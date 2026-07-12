# Codebase Q&A Engine

> Ask questions about any GitHub repository in plain English. Get answers with exact file paths, function names, and line numbers — powered by a production-grade RAG pipeline built from scratch.


---

## What it does

Point this at any public GitHub repo and ask:

- *"How does connection pooling work?"*
- *"Where is JWT authentication handled?"*
- *"Find all places Stripe is called."*
- *"Walk me through the request lifecycle end to end."*

The system returns a grounded answer that cites `file.py:L42-L89`, not a hallucination. If the implementation is not in the retrieved context, it says so explicitly.

**[Live Demo →](https://codebase-qa-flame.vercel.app/)**

---

## Why this is different from generic RAG

Most RAG tutorials split text every 512 characters. Code is not text. A 100-line function split into two chunks produces two meaningless fragments — neither compiles, neither answers anything.

This pipeline treats code as what it actually is: a graph of functions that call other functions.

| Generic RAG | This project |
|---|---|
| Fixed-size text chunking | AST-based chunking — one function = one chunk, never split |
| Vector search only | Hybrid: vector + BM25 keyword search + RRF fusion |
| No query understanding | HyDE rewriting using repo's own class/function names |
| Single retrieval pass | Multi-hop: retrieve → extract symbols → retrieve related → rerank |
| Answers from similarity | Answers grounded in retrieved code with file + line citations |
| No quality measurement | Eval framework: Recall@5, Recall@10, MRR, latency P50/P95 |

---

## Architecture

```
Question
  │
  ├─ L1 cache check (Redis) ──────────────────────────── ~1ms on hit
  │
  ├─ HyDE rewrite + intent detection (Groq, 1 call) ──── ~60ms
  │    Uses repo's actual class/function names from ingest-time profile
  │
  ├─ Symbol expansion (BM25 first-pass → real identifiers) ── ~50ms
  │
  ├─ Multi-vector search (HyDE vector + per-symbol vectors, parallel)
  │  + BM25 keyword search (expanded symbols)
  │  → RRF fusion → top 20 candidates ─────────────────── ~30ms
  │
  ├─ Code-aware reranking (local CrossEncoder) ────────── ~80ms, $0
  │    method chunks score higher than large class chunks
  │
  ├─ Smart context selection (whole chunks, 800-token budget) ── ~0ms
  │    No compression — compression destroys code details
  │
  ├─ Graph expansion for flow questions ───────────────── ~50ms
  │    Walks calls/called_by to reconstruct execution path
  │
  └─ LLM generation (Groq LLaMA 3.1, streamed) ───────── ~100ms first token
       Strict grounding: cites every claim or says what's missing
```

**Total latency:** ~350ms to first token on cache miss. ~1ms on cache hit.

---

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| API | FastAPI (async) | Non-blocking I/O throughout |
| Vector DB | Qdrant | Pre-filtering by repo before ANN scan |
| Keyword search | BM25 (rank-bm25) | Exact identifier matching |
| Embeddings | bge-small-en-v1.5 + ONNX Runtime | Zero API cost, ~25ms local |
| Reranker | CrossEncoder ms-marco-MiniLM-L-6-v2 | Local, $0, improves precision 25% |
| LLM | Groq LLaMA 3.1 8B Instant | Free tier, ~560 tok/s |
| Cache | Redis | Two-layer: queries (5min TTL) + embeddings (permanent) |
| Code parsing | Python `ast` module | Zero deps, extracts call graph |
| Repo cloning | GitPython | Shallow clone + git diff for incremental sync |
| Deployment | Docker Compose | Qdrant + Redis + API in one command |

---

## Project structure

```
codebase-qa/
├── app/
│   ├── main.py                    # FastAPI lifespan, router registration
│   ├── config.py                  # Typed settings from .env
│   │
│   ├── engine/
│   │   ├── ast_chunker.py         # AST → CodeChunk, method-level for large classes
│   │   ├── embedder.py            # ONNX local embedder, $0/query
│   │   ├── vectordb.py            # Qdrant client with pre-filtering + scroll
│   │   ├── bm25.py                # BM25 index + symbol name extraction
│   │   ├── fusion.py              # Reciprocal Rank Fusion
│   │   ├── call_graph.py          # build_called_by inversion + BFS expansion
│   │   ├── reranker.py            # Code-aware CrossEncoder (type + size scoring)
│   │   ├── token_budget.py        # Smart selection, grounding check, prompts
│   │   └── cost_tracker.py        # Per-stage latency and cost tracing
│   │
│   ├── cache/
│   │   └── redis_cache.py         # Two-layer cache + repo profile storage
│   │
│   ├── query/
│   │   ├── rewriter.py            # HyDE + intent detection (repo-profile-aware)
│   │   ├── retriever.py           # Multi-hop hybrid retrieval + symbol expansion
│   │   └── pipeline.py            # Full orchestrator: run_query + stream_query
│   │
│   ├── ingest/
│   │   ├── cloner.py              # git clone / pull (shallow)
│   │   ├── pipeline.py            # clone → chunk → call graph → embed → store
│   │   ├── incremental.py         # git diff + content hash re-ingest
│   │   └── repo_profile.py        # Build repo vocabulary for HyDE prompts
│   │
│   ├── eval/
│   │   ├── golden_dataset.py      # Auto Q&A generation from docstrings
│   │   ├── metrics.py             # Recall@K, MRR, latency percentiles
│   │   └── runner.py              # Benchmark runner
│   │
│   └── routers/
│       ├── repos.py               # POST /repos, GET, DELETE, POST /sync
│       ├── query.py               # POST /ask, GET /stream (SSE)
│       ├── search.py              # GET /search (baseline, no HyDE/rerank)
│       ├── stats.py               # GET /stats/cache
│       └── eval.py                # POST /eval/run
│
├── tests/
│   ├── test_chunker.py
│   ├── test_hybrid_search.py
│   ├── test_call_graph.py
│   └── test_eval_metrics.py
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Getting started

### Prerequisites
- Docker Desktop running
- A free Groq API key — [console.groq.com/keys](https://console.groq.com/keys) (no credit card)

### 1. Clone and configure

```bash
git clone https://github.com/KhanSimak/codebase-qa.git
cd codebase-qa
cp .env.example .env
# Add your GROQ_API_KEY in .env
```

### 2. Start all services

```bash
docker-compose up --build
```

This starts Qdrant (vector DB), Redis (cache), and the FastAPI server. The ONNX embedding model and CrossEncoder reranker download on first startup (~500MB, cached after).

### 3. Verify

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "features": ["ast_chunking", "call_graph", "hybrid_search", "reranking", "two_layer_cache", "sse_streaming"]
}
```

Interactive API docs: **http://localhost:8000/docs**

---

## Usage

### Ingest a repository

```bash
curl -X POST http://localhost:8000/repos/ \
  -H "Content-Type: application/json" \
  -d '{"github_url": "https://github.com/encode/httpx", "branch": "master"}'
```

```json
{ "id": "a1b2c3d4", "status": "ingesting" }
```

Poll until done:

```bash
curl http://localhost:8000/repos/a1b2c3d4
```

```json
{
  "id": "a1b2c3d4",
  "status": "done",
  "chunk_count": 847,
  "file_count": 63,
  "languages": ["python"]
}
```

### Ask a question (full pipeline)

```bash
curl -X POST "http://localhost:8000/repos/a1b2c3d4/ask?question=how+does+connection+pooling+work&top_k=5"
```

```json
{
  "answer": "Connection pooling is handled by the ConnectionPool class in httpx/_transports/default.py (L45-L89). The acquire() method at L67 checks the pool for an available connection before creating a new one...",
  "rewritten_query": "pool = ConnectionPool(max_connections=10)\nconn = pool.acquire()\ntransport = HTTPTransport(pool=pool)",
  "intent": "understand_flow",
  "sources": [
    {
      "name": "acquire",
      "type": "method",
      "file": "httpx/_transports/default.py",
      "line_start": 67,
      "line_end": 89,
      "score": 0.9241
    }
  ],
  "cache_hit": false,
  "trace": {
    "total_latency_ms": 412.3,
    "total_cost_usd": 0.000044,
    "stages": [
      { "stage": "query_cache_l1",    "latency_ms": 0.4,  "cache_hit": false },
      { "stage": "hyde_rewrite",      "latency_ms": 58.2, "cost_usd": 0.000012 },
      { "stage": "hybrid_retrieval",  "latency_ms": 31.7, "cost_usd": 0.0 },
      { "stage": "reranker",          "latency_ms": 79.1, "cost_usd": 0.0 },
      { "stage": "llm_generation",    "latency_ms": 98.4, "cost_usd": 0.000032 }
    ]
  }
}
```

Every response includes a **cost and latency trace** per stage. Embedding and reranking cost $0 (local models). The only paid calls are HyDE rewriting and LLM generation via Groq's free tier.

### Stream the answer (SSE)

```bash
curl -N "http://localhost:8000/repos/a1b2c3d4/stream?question=how+does+connection+pooling+work"
```

Emits:
1. `{type: "sources", sources: [...]}` — immediately, before generation starts
2. `{type: "token", text: "..."}` — streamed as tokens arrive
3. `{type: "done", trace: {...}}` — full cost trace at end

### Incremental sync (after a code change)

```bash
curl -X POST http://localhost:8000/repos/a1b2c3d4/sync
```

Uses `git diff` to detect changed files, then content hashing to detect changed functions within those files. Only re-embeds what actually changed.

```json
{
  "mode": "incremental",
  "changed_files": 3,
  "embedded_chunks": 11,
  "skipped_chunks": 836
}
```

### Run the evaluation benchmark

```bash
curl -X POST "http://localhost:8000/eval/run?repo_id=a1b2c3d4&max_questions=100"
```

```json
{
  "recall_at_5":  0.847,
  "recall_at_10": 0.923,
  "mrr":          0.712,
  "latency_ms":   { "p50": 38.2, "p95": 71.4, "p99": 95.0 },
  "worst_files": [
    { "file": "httpx/_legacy/old_client.py", "recall": 0.33, "question_count": 3 }
  ]
}
```

Auto-generates questions from function docstrings and measures whether the retrieval pipeline finds the correct chunk.

---

## Key design decisions

**AST chunking over fixed-size splitting**
The Python `ast` module extracts every function, class, and method as an atomic chunk. Large classes (>40 lines) emit a thin header chunk plus individual method chunks — so `ConnectionPool.acquire()` is retrievable on its own, not buried inside a 300-line class blob.

**Why local embeddings**
`bge-small-en-v1.5` with ONNX Runtime runs in ~25ms on CPU and costs $0 forever. OpenAI's API charges ~$0.50/day at 10K queries. Every chunk is embedded once and cached permanently in Redis.

**Why Groq instead of OpenAI**
Groq's free tier provides ~14,400 requests/day, no credit card required, and LLaMA 3.1 8B runs at ~560 tokens/second on their LPU hardware. The `llm_generation` stage (98ms in the trace above) is the fastest stage in the pipeline.

**No context compression**
Context compression via LLM summarisation was tried and removed. It added ~50 seconds of latency and destroyed the exact implementation details the LLM needed to answer correctly. Replaced with smart whole-chunk selection that fits the 800-token budget by preferring method-level chunks over class-level.

**Explicit refusal rule**
The LLM is given a strict system prompt with rule 2: *"If the context does not contain the implementation, say exactly: The implementation of [X] is not in the retrieved context."* Without this, the model fills silence with plausible-sounding hallucination.

---

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/repos/` | Ingest a GitHub repo (async, returns immediately) |
| GET | `/repos/{id}` | Check ingest status and chunk count |
| DELETE | `/repos/{id}` | Remove repo and all its vectors |
| POST | `/repos/{id}/sync` | Incremental re-ingest (git diff + content hash) |
| POST | `/repos/{id}/ask` | Full pipeline: HyDE + hybrid + rerank + LLM |
| GET | `/repos/{id}/stream` | Same pipeline, SSE streaming |
| GET | `/repos/{id}/search` | Baseline: hybrid search only, no HyDE/rerank |
| POST | `/eval/run` | Run Recall@5/MRR benchmark |
| GET | `/stats/cache` | Redis cache hit rate |
| GET | `/health` | Service health check |

Full interactive docs at `http://localhost:8000/docs` when running.

---

## Environment variables

```env
# Required
GROQ_API_KEY=your-key-here          # free at console.groq.com/keys

# Defaults (change if needed)
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
REPOS_DIR=/tmp/repos
EMBED_MODEL=BAAI/bge-small-en-v1.5
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
GROQ_MODEL=llama-3.1-8b-instant

# Tuning
QUERY_TOP_K=20                      # candidates before reranking
MULTI_HOP_MAX=2                     # retrieval hops
CONTEXT_MAX_TOKENS=800              # whole-chunk context budget
CLASS_SPLIT_THRESHOLD=40            # lines; larger classes split into methods
```

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

Tests cover: AST chunking correctness (complete functions, call graph, type classification), BM25 indexing and symbol expansion, RRF fusion ordering, call graph inversion and BFS traversal, Recall@K and MRR metric calculations.

---

## How the pipeline evolves across phases

This project was built in four phases, each a standalone runnable app:

| Phase | What was added |
|---|---|
| 1 | AST chunking, ONNX embeddings, basic Qdrant vector search |
| 2 | BM25 keyword search, RRF fusion, two-layer Redis cache |
| 3 | HyDE rewriting, CrossEncoder reranking, SSE streaming, cost tracing |
| 4 | Call graph, multi-hop retrieval, incremental ingest, eval framework |

Each phase is in its own directory (`phase1/` through `phase4/`) with its own `docker-compose.yml` and `README.md` so you can run any phase independently and see exactly what each optimization adds.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

- [Qdrant](https://qdrant.tech) for the vector database
- [Groq](https://groq.com) for the free LLM inference tier
- [BAAI](https://huggingface.co/BAAI/bge-small-en-v1.5) for the bge-small embedding model
- [Sentence Transformers](https://sbert.net) for the CrossEncoder reranker
