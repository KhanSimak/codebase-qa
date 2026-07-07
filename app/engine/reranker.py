from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import asyncio

_model = None
_executor = ThreadPoolExecutor(max_workers=2)

def load_embedder(cfg):
    global _model
    _model = SentenceTransformer(cfg.embed_model)
    return _model

async def embed_text(text, is_query=True):
    if is_query:
        text = f"Represent this sentence: {text}"

    loop = asyncio.get_running_loop()

    vec = await loop.run_in_executor(
        _executor,
        lambda: _model.encode(text, normalize_embeddings=True)
    )

    return vec.tolist()

async def embed_batch(texts):
    if not texts:
        return []

    loop = asyncio.get_running_loop()

    vectors = await loop.run_in_executor(
        _executor,
        lambda: _model.encode(texts, normalize_embeddings=True)
    )

    return vectors.tolist()