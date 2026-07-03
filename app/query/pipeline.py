"""
pipeline.py — the complete query pipeline, fully traced

Stage order (every stage wrapped in a StageTimer, see cost_tracker.py):
  1. L1 cache check (exact question+repo+filters match)         -> ~1ms on hit
  2. HyDE rewrite + intent detection (one Groq call)             -> ~60ms
  3. Embed the HyDE snippet (L2 cache checked first)             -> ~25ms or ~1ms cached
  4. Hybrid retrieval: vector + BM25 + RRF (+ graph expand)      -> ~25-60ms
  5. Cross-encoder rerank: top 20 -> top 5                       -> ~80ms, $0
  6. Parallel context compression + token budget enforcement     -> ~200ms
  7. Final LLM call (llama-3.1-8b-instant via Groq), streamed or not -> Groq's
     LPU hardware makes this the fastest stage in the whole pipeline, not
     the slowest — first tokens typically arrive well under 150ms.
  8. Cache the result for next time

Both a blocking `run_query` (returns the full answer + trace) and a
streaming `stream_query` (SSE generator) are provided — same pipeline,
different output shape.

NOTE ON STREAMING SHAPE: Anthropic's SDK provides a `.messages.stream()`
async context manager with a convenience `.text_stream` iterator. Groq's
SDK is OpenAI-style instead — `stream=True` on the normal `.create()` call
returns an async-iterable stream directly, and each chunk's text lives at
`chunk.choices[0].delta.content` (which can be `None` for the very first
chunk and the final chunk, so it's guarded below).
"""

import json
import logging
from groq import AsyncGroq

from app.engine.cost_tracker import RequestTrace
from app.engine.reranker import rerank
from app.engine.token_budget import  select_context, build_prompt, count_tokens
from app.query.rewriter import rewrite_query, GRAPH_EXPAND_INTENTS
from app.query.retriever import retrieve
from app.cache.redis_cache import get_cached_query, set_cached_query
from app.config import get_settings

settings = get_settings()

_llm = AsyncGroq(
    api_key=settings.groq_api_key
)


async def _do_retrieval_and_rerank(question: str, repo_id: str, qdrant_client, redis_client, cfg, top_k: int, trace: RequestTrace):
    """Shared by both run_query and stream_query — stages 2 through 5."""

    stage_rewrite = trace.start_stage("hyde_rewrite")
    rewrite = await rewrite_query(question,repo_id,redis_client)
    stage_rewrite.input_tokens  = count_tokens(question) + 150   # prompt template overhead, approx
    stage_rewrite.output_tokens = count_tokens(json.dumps(rewrite))
    stage_rewrite.finish()

    graph_expand = rewrite["intent"] in GRAPH_EXPAND_INTENTS

    stage_retrieve = trace.start_stage("hybrid_retrieval" + ("_graph_expanded" if graph_expand else ""))
    candidates = await retrieve(
        question=question, hyde_snippet = rewrite["implementation_summary"], phrases=rewrite["phrases"],
          intent=rewrite["intent"], repo_id=repo_id, qdrant_client=qdrant_client,
        redis_client=redis_client, cfg=cfg, top_k=cfg.query_top_k, graph_expand=graph_expand,
    )
    stage_retrieve.finish()

    if not candidates:
        return [], rewrite

    stage_rerank = trace.start_stage("reranker")
    reranked = await rerank(question, candidates, top_n=10)
    print("=" * 60)
    print("RERANKED")

    for c in reranked:
     print(
        c["name"],
        c["type"],
        c.get("rerank_score"),
    )
    stage_rerank.finish()

    return reranked, rewrite


async def run_query(question: str, repo_id: str, qdrant_client, redis_client, cfg, top_k: int = 5) -> dict:
    trace = RequestTrace(query=question, repo_id=repo_id)

    # ── Stage 1: L1 cache ────────────────────────────────────────────────
    stage_cache = trace.start_stage("query_cache_l1")
    cached = await get_cached_query(redis_client, repo_id, question, top_k)
    stage_cache.cache_hit = cached is not None
    stage_cache.finish()
    if cached:
        return {**cached, "cache_hit": True, "trace": trace.summary()}

    reranked, rewrite = await _do_retrieval_and_rerank(question, repo_id, qdrant_client, redis_client, cfg, top_k, trace)

    if not reranked:
        return {
            "question": question, "answer": "No relevant code found in this repository.",
            "sources": [], "cache_hit": False, "trace": trace.summary(),
        }

    # ── Compress + budget ────────────────────────────────────────────────
    stage_select = trace.start_stage("context_selection")
# Keep the top-N ranked chunks regardless of score.
# CrossEncoder scores are relative, not absolute.

    final_chunks = select_context(reranked)
    print("=" * 60)
    print("FINAL CONTEXT")

    for c in final_chunks:
     print(
        c["name"],
        c["type"],
        c.get("rerank_score"),
    )
    stage_select.finish()

    # ── Final LLM call ───────────────────────────────────────────────────
    stage_llm = trace.start_stage("llm_generation")
    system_prompt, user_msg = build_prompt(
    question,
    final_chunks,
    rewrite["intent"],
)
    stage_llm.input_tokens = count_tokens(system_prompt) + count_tokens(user_msg)

    response = await _llm.chat.completions.create(
        model=settings.groq_model,
        max_completion_tokens=400,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    answer = response.choices[0].message.content
    stage_llm.output_tokens = count_tokens(answer)
    stage_llm.finish()

    result = {
        "question": question,
        "answer":   answer,
        "rewritten_query": rewrite["implementation_summary"],
        "intent":   rewrite["intent"],
        "sources": [
            {
                "id": c["id"], "name": c.get("name", ""), "type": c.get("type", ""),
                "file": c.get("file", ""), "line_start": c.get("line_start", 0),
                "line_end": c.get("line_end", 0), "score": c.get("rerank_score", c.get("score", 0.0)),
            }
            for c in reranked
        ],
        "cache_hit": False,
    }
    await set_cached_query(redis_client, repo_id, question, top_k, result)
    return {**result, "trace": trace.summary()}


async def stream_query(question: str, repo_id: str, qdrant_client, redis_client, cfg, top_k: int = 5):
    """SSE generator. Emits sources BEFORE the LLM starts, then streams tokens."""
    trace = RequestTrace(query=question, repo_id=repo_id)

    cached = await get_cached_query(redis_client, repo_id, question, top_k)
    if cached:
        yield f"data: {json.dumps({'type':'sources','sources':cached.get('sources',[]),'cached':True})}\n\n"
        yield f"data: {json.dumps({'type':'token','text':cached['answer']})}\n\n"
        yield f"data: {json.dumps({'type':'done','trace':trace.summary()})}\n\n"
        return

    reranked, rewrite = await _do_retrieval_and_rerank(question, repo_id, qdrant_client, redis_client, cfg, top_k, trace)

    if not reranked:
        yield f"data: {json.dumps({'type':'error','text':'No relevant code found.'})}\n\n"
        return

    sources = [
        {"id": c["id"], "name": c.get("name"), "file": c.get("file"),
         "line_start": c.get("line_start"), "line_end": c.get("line_end")}
        for c in reranked
    ]
    yield f"data: {json.dumps({'type':'sources','sources':sources,'rewrite':rewrite['implementation_summary'],'intent':rewrite['intent']})}\n\n"

    stage_select = trace.start_stage("context_selection")

    final_chunks = select_context(reranked, max_tokens=800)

    stage_select.finish()
    system_prompt, user_msg = build_prompt(question, final_chunks)

    stage_llm = trace.start_stage("llm_generation")
    stage_llm.input_tokens = count_tokens(system_prompt) + count_tokens(user_msg)

    full_answer = []
    stream = await _llm.chat.completions.create(
        model=settings.groq_model,
        max_completion_tokens=400,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:   # the first chunk (role announcement) and last chunk often have None content
            full_answer.append(delta)
            yield f"data: {json.dumps({'type':'token','text':delta})}\n\n"

    answer = "".join(full_answer)
    stage_llm.output_tokens = count_tokens(answer)
    stage_llm.finish()

    result = {"question": question, "answer": answer, "sources": sources, "intent": rewrite["intent"]}
    await set_cached_query(redis_client, repo_id, question, top_k, result)

    yield f"data: {json.dumps({'type':'done','trace':trace.summary()})}\n\n"
