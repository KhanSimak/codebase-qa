"""
reranker.py — cross-encoder reranking, local and free

BI-ENCODER (what Qdrant/embedder.py does) vs CROSS-ENCODER (this file):
  Bi-encoder: embed the query, embed the document, SEPARATELY, then compare
  with a dot product. Fast (you can pre-compute document embeddings once),
  but the model never actually sees the query and document together —
  it's comparing two independent summaries.

  Cross-encoder: feed (query, document) into the model TOGETHER as one
  input. The model's attention layers can directly relate specific words
  in the query to specific words in the document. Far more accurate, but
  too slow to run against every chunk in a large repo — which is exactly
  why we use bi-encoder search FIRST to narrow thousands of chunks down
  to ~20 candidates, then cross-encoder reranking to pick the real top 5.

COST: $0. ms-marco-MiniLM-L-6-v2 is ~130MB, runs on CPU in ~80ms for
20 pairs, no API call involved.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_reranker = None
_executor = ThreadPoolExecutor(max_workers=2)


def load_reranker(cfg):
    global _reranker
    logger.info(f"Loading reranker: {cfg.rerank_model}")
    _reranker = CrossEncoder(cfg.rerank_model)
    logger.info("Reranker ready — $0/query, local CPU inference")
    return _reranker

def _rerank_sync(question: str, chunks: list[dict], top_n: int) -> list[dict]:
    if not chunks:
        return []
    # CrossEncoder has a 512-token input limit — truncate rather than skip,
    # partial context still beats throwing the candidate away entirely.
    pairs  = [(question, chunk["text"][:512]) for chunk in chunks]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [{**chunk, "rerank_score": round(float(score), 4)} for chunk, score in ranked[:top_n]]


async def rerank(question: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    if not chunks:
        return []
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _rerank_sync, question, chunks, top_n)
