"""
call_graph.py — building and walking the codebase's call graph

THE CENTRAL IDEA OF THIS WHOLE PHASE:
  A codebase is not a bag of independent chunks — it's a graph. Functions
  call other functions. "How does payment processing work?" is not answered
  by ONE function; it's answered by an entry point PLUS everything it calls
  PLUS (often) everything that calls it. Phases 1-3 only ever return isolated
  chunks. This file is what turns "find a function" into "understand a flow."

STEP 1 — INVERSION (calls -> called_by)
  Every chunk already knows what it calls (extracted by ast_chunker.py at
  parse time — that's a per-function, local operation). called_by is the
  GLOBAL inverse of that: "who calls me?" can only be answered once you've
  seen every chunk in the repo, because the caller could be in any file.
  So inversion happens once, at the END of ingest, across all chunks together.

STEP 2 — MATCHING CAVEAT (read this before you trust the graph too much)
  `calls` stores bare names extracted from the AST — e.g. a call like
  `self._get_user(x)` is recorded as "_get_user", not as a fully qualified
  symbol. When we invert, we match callee names against chunk NAMES across
  the whole repo. This means two unrelated classes that both happen to
  define a method called `save()` will be treated as the same callee.
  This is a deliberate, honest simplification — full symbol resolution
  would need a real type checker (or something like Jedi/LSP). For RAG
  retrieval purposes, "approximately right call graph" still meaningfully
  improves flow-style answers; it does not need to be a perfect compiler.

STEP 3 — GRAPH EXPANSION AT QUERY TIME
  Given an entry chunk, walk `calls` (downstream/callees) and/or `called_by`
  (upstream/callers) outward to a fixed depth using BFS. This is what lets
  "explain the payment flow" return the entry point AND its immediate
  collaborators, instead of just the one function that ranked #1.
"""

from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


def build_called_by(all_chunks: list) -> None:
    """
    Mutates each CodeChunk in `all_chunks` in place, filling in `called_by`.

    O(N * avg_calls_per_chunk) — for a repo with a few thousand chunks and
    each chunk calling ~5 other functions on average, this is a few
    thousand dict insertions. Fast, done once per ingest.
    """
    name_to_chunks: dict[str, list] = {}
    for chunk in all_chunks:
        name_to_chunks.setdefault(chunk.name, []).append(chunk)

    for chunk in all_chunks:
        for callee_name in chunk.calls:
            for callee_chunk in name_to_chunks.get(callee_name, []):
                if chunk.name not in callee_chunk.called_by:
                    callee_chunk.called_by.append(chunk.name)

    total_edges = sum(len(c.called_by) for c in all_chunks)
    logger.info(f"Call graph built: {len(all_chunks)} nodes, {total_edges} called_by edges")


def expand_by_graph(
    entry_chunks: list[dict],
    name_index: dict[str, list[dict]],
    depth: int = 2,
    direction: str = "both",   # "callees" | "callers" | "both"
    max_expanded: int = 15,
) -> list[dict]:
    """
    BFS outward from a set of entry chunks (the top retrieval results),
    following `calls` (callees) and/or `called_by` (callers) edges.

    entry_chunks: the chunks retrieval already found — these are ALWAYS kept.
    name_index:   {chunk_name: [chunk_dict, ...]} for the whole repo, built
                  once per query from whatever set of chunks you have on hand
                  (usually all chunks in the repo, fetched once and reused).
    Returns:      entry_chunks + everything discovered by the graph walk,
                  deduplicated by id, capped at max_expanded ADDITIONAL chunks
                  so a hub function with hundreds of callers doesn't blow up
                  the context window.
    """
    visited_ids = {c["id"] for c in entry_chunks}
    result      = list(entry_chunks)
    frontier    = list(entry_chunks)

    for _ in range(depth):
        next_frontier = []
        for chunk in frontier:
            neighbor_names = []
            if direction in ("callees", "both"):
                neighbor_names.extend(chunk.get("calls", []))
            if direction in ("callers", "both"):
                neighbor_names.extend(chunk.get("called_by", []))

            for name in neighbor_names:
                for neighbor in name_index.get(name, []):
                    if neighbor["id"] not in visited_ids:
                        visited_ids.add(neighbor["id"])
                        result.append(neighbor)
                        next_frontier.append(neighbor)
                        if len(result) - len(entry_chunks) >= max_expanded:
                            return result
        frontier = next_frontier
        if not frontier:
            break

    return result


def build_name_index(chunks: list[dict]) -> dict[str, list[dict]]:
    """Helper: group a repo's chunk dicts by name, for graph expansion lookups."""
    index: dict[str, list[dict]] = {}
    for chunk in chunks:
        index.setdefault(chunk["name"], []).append(chunk)
    return index
