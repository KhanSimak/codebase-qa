"""
retriever.py — hybrid retrieval, with optional call-graph expansion

Combines everything built so far:
  1. Embed the HyDE snippet (with Redis cache check)
  2. Vector search (Qdrant) + BM25 search, run together
  3. RRF fusion -> top 20 candidates
  4. IF the question's intent benefits from it (understand_flow, find_usage,
     debug — see rewriter.py), expand the candidate set by walking the call
     graph outward from the fused results, using the called_by/calls edges
     built at ingest time (app/engine/call_graph.py)

Graph expansion needs the WHOLE repo's chunk dicts in memory to build a
name index (see call_graph.build_name_index). For a repo of a few thousand
chunks this is a single Qdrant scroll call and is cheap; we don't re-fetch
it per request in a hot loop — production code would cache this per repo_id,
which is exactly the kind of thing the Redis layer here could be extended
to do (left as a natural next optimization beyond this project's scope).
"""

import logging
from app.engine.embedder import embed_text
from app.engine.vectordb import search as vector_search, retrieve_by_ids, scroll_repo_chunks
from app.engine.bm25 import search as bm25_search
from app.engine.fusion import reciprocal_rank_fusion
from app.engine.call_graph import expand_by_graph, build_name_index
from app.cache.redis_cache import get_cached_embedding, set_cached_embedding
from app.query.symbol_expander import expand_to_symbols

logger = logging.getLogger(__name__)


async def retrieve(
    question: str,
    hyde_snippet: str,
    phrases: list[str],
    intent: str,
    repo_id: str,
    qdrant_client,
    redis_client,
    cfg,
    top_k: int = 20,
    graph_expand: bool = False,
) -> list[dict]:
    """Returns a list of chunk dicts, fused and (optionally) graph-expanded."""
    
    # ── Embed the HyDE snippet, checking cache first ────────────────────────
    cached_question = await get_cached_embedding(redis_client, question)

    if cached_question:
      question_vector = cached_question
    else:
      question_vector = await embed_text(question, is_query=True)
    await set_cached_embedding(redis_client, question, question_vector)

    cached_hyde = await get_cached_embedding(redis_client, hyde_snippet)

    if cached_hyde:
      hyde_vector = cached_hyde
    else:
      hyde_vector = await embed_text(hyde_snippet, is_query=True)
    await set_cached_embedding(redis_client, hyde_snippet, hyde_vector)


    symbols = await expand_to_symbols(
    phrases,
    repo_id,
)
    # ── Vector + BM25 ────────────────────────────────────────────────────────
    question_hits = await vector_search(
      qdrant_client,
      cfg.qdrant_collection,
      question_vector,
      repo_id,
      top_k=top_k,
    )

    hyde_hits = await vector_search(
      qdrant_client,
      cfg.qdrant_collection,
      hyde_vector,
      repo_id,
      top_k=top_k,
    )

    keyword_hits: dict[str, dict] = {}
    search_terms = [question] + phrases + symbols
    search_terms = list(dict.fromkeys(search_terms))

    for phrase in search_terms:
        for hit in bm25_search(repo_id, phrase, top_k=25):
            doc_id = hit["id"]
            if doc_id not in keyword_hits or hit["bm25_score"] > keyword_hits[doc_id]["bm25_score"]:
                keyword_hits[doc_id] = hit
    keyword_hits_list = sorted(keyword_hits.values(), key=lambda x: x["bm25_score"], reverse=True)

    if not question_hits and not hyde_hits and not keyword_hits_list:
      return []

    # ── RRF fusion ───────────────────────────────────────────────────────────
    combined_vector_hits = question_hits + hyde_hits

    fused_ids = reciprocal_rank_fusion(
      combined_vector_hits,
      keyword_hits_list,
      top_k=top_k
    )

    vector_lookup = {}
    for hit in question_hits:
      vector_lookup[hit["id"]] = hit


    for hit in hyde_hits:
      if hit["id"] not in vector_lookup:
        vector_lookup[hit["id"]] = hit

     
    missing_ids   = [doc_id for doc_id in fused_ids if doc_id not in vector_lookup]
    fetched       = await retrieve_by_ids(qdrant_client, cfg.qdrant_collection, missing_ids) if missing_ids else {}

    fused_chunks = []
    for doc_id in fused_ids:
        chunk = vector_lookup.get(doc_id) or fetched.get(doc_id)
        if chunk:
            fused_chunks.append(chunk)
    

    q = question.lower()

    for chunk in fused_chunks:
     if chunk["name"].lower() == q:
        chunk["score"] += 100
    print("=" * 60)
    print("FUSED CHUNKS")

    for c in fused_chunks:
     print(
        c["name"],
        c["type"],
        round(c.get("score", 0), 4)
    )
    if not graph_expand:
        return fused_chunks

    # ── Graph expansion — only for intents that benefit from it ───────────
    logger.info(f"Graph-expanding from {len(fused_chunks)} entry chunks (intent triggered expansion)")
    all_repo_chunks = await scroll_repo_chunks(qdrant_client, cfg.qdrant_collection, repo_id)
    print("=" * 60)
    print("TOTAL REPO CHUNKS:", len(all_repo_chunks))

    if all_repo_chunks:
       print("Sample keys:", all_repo_chunks[0].keys())
    name_index = build_name_index(all_repo_chunks)
    expanded = expand_by_graph(
    fused_chunks,
    name_index,
    depth=2,
    direction="both",
    max_expanded=15,
    )
    print("Before:", len(fused_chunks))
    print("After :", len(expanded))
    print("=" * 60)
    print("GRAPH EXPANDED")

    for c in expanded:
     print(
        c["name"],
        c["type"]
    )

    print("Question vector hits:", len(question_hits))
    print("HyDE vector hits:", len(hyde_hits))
    print("BM25 hits:", len(keyword_hits_list))
    print("Fused IDs:", len(fused_ids))
    print("Fused chunks:", len(fused_chunks))
    print("Graph expanded:", len(expanded) if graph_expand else "Skipped")
    return expanded