from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import Settings, get_settings
from app.rag.step2_chunk import Chunk


# trun each chonk from step2 to the normalized vector form
@dataclass
class EmbeddingResult:
    """What Step 3 returns: model info + the vector matrix."""

    model: str                      # name of the embedding model used
    dim: int                        # vector length (384 for bge-small)
    vectors: np.ndarray             # shape (n_chunks, dim), float32, L2-normalized
    chunk_ids: list[str] = field(default_factory=list)  # row-aligned chunk ids


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    """Create/load a sentence-transformers model on CPU."""
    return SentenceTransformer(model_name, device="cpu")


def _texts_for_embedding(chunks: Sequence[Chunk]) -> list[str]:
    """Build the text that gets embedded for each chunk."""
    return [
        f"{chunk.chapter}\n{chunk.text}" if chunk.chapter else chunk.text
        for chunk in chunks
    ]


def _to_float32(array) -> np.ndarray:
    """Coerce an array-like to a float32 ndarray."""
    return np.asarray(array, dtype=np.float32)


def _normalize_or_check(vectors: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Ensure vectors are L2-normalized; re-normalize if needed."""
    vectors = _to_float32(vectors)
    if vectors.size == 0:
        return vectors

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.allclose(norms, 1.0, atol=tol):
        return vectors

    if np.any(norms == 0):
        raise ValueError("cannot normalize a zero-length embedding vector")

    return (vectors / norms).astype(np.float32, copy=False)


def load_embedder(model_name: str | None = None) -> SentenceTransformer:
    """Public helper for FastAPI lifespan: load model once at startup."""
    if model_name is None:
        model_name = get_settings().embedding_model
    return _load_model(model_name)


def embed_chunks(
    chunks: list[Chunk],
    settings: Settings,
    batch_size: int | None = None,
    model: SentenceTransformer | None = None,  # inject a fake in tests
) -> EmbeddingResult:
    """Embed chunks into normalized vectors."""
    if settings.embedding_backend != "sentence-transformers":
        raise ValueError(
            f"unsupported embedding_backend={settings.embedding_backend!r}; "
            "this step supports 'sentence-transformers'"
        )

    if not chunks:
        return EmbeddingResult(
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            vectors=np.empty((0, settings.embedding_dim), dtype=np.float32),
            chunk_ids=[],
        )

    batch_size = settings.embedding_batch if batch_size is None else batch_size
    if batch_size <= 0:
        raise ValueError("batch size must be greater than zero")

    if model is None:
        model = load_embedder(settings.embedding_model)

    raw = model.encode(
        _texts_for_embedding(chunks),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    vectors = _normalize_or_check(_to_float32(raw))

    expected_rows = len(chunks)
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
        raise ValueError(
            f"expected {expected_rows} vectors, got shape {vectors.shape}"
        )

    dim = vectors.shape[1]
    if dim != settings.embedding_dim:
        raise ValueError(
            f"embedding_dim={settings.embedding_dim} does not match model output "
            f"dim={dim}; update Settings.embedding_dim or re-embed the whole book"
        )

    return EmbeddingResult(
        model=settings.embedding_model,
        dim=dim,
        vectors=vectors,
        chunk_ids=[chunk.id for chunk in chunks],
    )
