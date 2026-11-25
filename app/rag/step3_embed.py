"""Step 3 - Embed: chunks -> normalized vectors.

Input comes from:
    Step 2: list[Chunk] from app.rag.step2_chunk.chunk_book().
    The text embedded is "chapter\\nchunk text" (see _texts_for_embedding).

Output:
    EmbeddingResult(model, dim, vectors, chunk_ids), where vectors is a
    float32 matrix of shape (n_chunks, dim), L2-normalized and row-aligned
    with chunk_ids.

Who consumes it:
    Step 4 (app.rag.step4_store.store_index) writes the vectors to SQLite.
    Step 5 reuses the same embedder for the query vector.

Important: the embedding model is frozen. Index-time and query-time must use
the same model/dimension or their vectors are not comparable.
"""

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
    """Build the text that gets embedded for each chunk.

    Input comes from Step 2 via embed_chunks(). Prepending the chapter title
    gives the vector a little topical context; Step 5 shows the original
    Chunk.text, not this prefixed version.
    """
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
    """Public helper for FastAPI lifespan: load model once at startup.

    Also called by Step 5 (_embed_question) so the query is embedded with
    exactly the same model used for the stored chunks.
    """
    if model_name is None:
        model_name = get_settings().embedding_model
    return _load_model(model_name)


def embed_chunks(
    chunks: list[Chunk],
    settings: Settings,
    batch_size: int | None = None,
    model: SentenceTransformer | None = None,  # inject a fake in tests
) -> EmbeddingResult:
    """Embed chunks into normalized vectors.

    Input comes from Step 2 (chunk_book). The returned EmbeddingResult is
    passed to Step 4 (store_index), which validates row alignment.

    Args:
        chunks: Chunks produced by Step 2.
        settings: App settings (backend, model name, dimension, batch size).
        batch_size: Optional override of settings.embedding_batch.
        model: Optional preloaded/fake encoder; loaded from settings if None.

    Returns:
        EmbeddingResult with one vector per chunk, L2-normalized.

    Raises:
        ValueError: Unsupported backend, bad batch size, wrong row count, or
            a dimension that does not match settings.embedding_dim.
    """
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
