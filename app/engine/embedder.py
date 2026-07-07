import asyncio
import logging
import os

from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

_executor = ThreadPoolExecutor(max_workers=4)

_model = None


def load_embedder(cfg):
    global _model

    logger.info(f"Loading {cfg.embed_model}")

    _model = SentenceTransformer(cfg.embed_model)

    logger.info("Embedder loaded")

    return _model


async def embed_text(text, is_query=True):

    if is_query:
        text = f"Represent this sentence: {text}"

    loop = asyncio.get_running_loop()

    vector = await loop.run_in_executor(
        _executor,
        lambda: _model.encode(
            text,
            normalize_embeddings=True
        )
    )

    return vector.tolist()


async def embed_batch(texts):

    if not texts:
        return []

    loop = asyncio.get_running_loop()

    vectors = await loop.run_in_executor(
        _executor,
        lambda: _model.encode(
            texts,
            normalize_embeddings=True
        )
    )

    return vectors.tolist()