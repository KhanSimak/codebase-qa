"""
query.py — the final pipeline's public API

POST /repos/{id}/ask     — full pipeline, blocking, returns a cost/latency trace
GET  /repos/{id}/stream  — same pipeline, SSE streaming

This supersedes search.py's /search endpoint for anything that needs
HyDE rewriting, reranking, token budgeting, graph expansion, or a cost
trace. /search (Phase 2) is left in place as the simpler baseline you can
diff against — hit both with the same question and compare the `sources`
and latency to see exactly what each added stage changes.
"""

from fastapi import APIRouter, Request, HTTPException, Query as QParam
from fastapi.responses import StreamingResponse

from app.query.pipeline import run_query, stream_query
from app.routers.repos import get_registry

router = APIRouter()


@router.post("/{repo_id}/ask")
async def ask(
    repo_id: str,
    request: Request,
    question: str = QParam(..., min_length=3),
    top_k: int = QParam(default=5, ge=1, le=10),
):
    registry = get_registry()
    if repo_id not in registry:
        raise HTTPException(404, "Repo not found")
    if registry[repo_id]["status"] != "done":
        raise HTTPException(400, f"Repo not ready: {registry[repo_id]['status']}")

    cfg, qdrant, redis_client = request.app.state.settings, request.app.state.qdrant, request.app.state.redis
    return await run_query(question, repo_id, qdrant, redis_client, cfg, top_k=top_k)


@router.get("/{repo_id}/stream")
async def ask_stream(
    repo_id: str,
    request: Request,
    question: str = QParam(..., min_length=3),
    top_k: int = QParam(default=5, ge=1, le=10),
):
    """
    SSE events emitted, in order:
      {"type":"sources","sources":[...],"rewrite":"...","intent":"..."}
      {"type":"token","text":"..."}  (repeated as tokens stream in)
      {"type":"done","trace":{...}}
    """
    registry = get_registry()
    if repo_id not in registry:
        raise HTTPException(404, "Repo not found")
    if registry[repo_id]["status"] != "done":
        raise HTTPException(400, "Repo not ready")

    cfg, qdrant, redis_client = request.app.state.settings, request.app.state.qdrant, request.app.state.redis
    return StreamingResponse(
        stream_query(question, repo_id, qdrant, redis_client, cfg, top_k=top_k),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
