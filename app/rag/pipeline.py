"""Indexing pipeline: runs Steps 1-4 in order.

Input : a book file path plus an open SQLAlchemy session.
Flow  :
    Step 1 ingest_book()   -> Book (chapters + page metadata)
    Step 2 chunk_book()    -> list[Chunk]
    Step 3 embed_chunks()  -> EmbeddingResult (vectors)
    Step 4 store_index()   -> rows in the ``book_rag`` Postgres schema

Step 5 (answer_question) is the read path and is NOT part of this
pipeline; it runs per query against the index written here.

Transaction rule: this module only flushes. The caller owns the session and
therefore owns commit/rollback. Building the engine and creating the schema
is the composition root's job (``app.main`` lifespan or a CLI entrypoint),
never this module's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.rag.step1_ingest import Book, ingest_book
from app.rag.step2_chunk import Chunk, chunk_book
from app.rag.step3_embed import EmbeddingResult, embed_chunks
from app.rag.step4_store import (
    StoredIndex,
    get_document_id,
    reset_index,
    store_index,
)


@dataclass
class IndexResult:
    """Everything produced while indexing one book.

    Attributes:
        book: Step 1 output (metadata + cleaned chapters).
        chunks: Step 2 output, in book order.
        embedded: Step 3 output, row-aligned with ``chunks``.
        stored: Step 4 output describing what was written.
        replaced: True when an older copy of the same sha256 was removed.
    """

    book: Book
    chunks: list[Chunk]
    embedded: EmbeddingResult
    stored: StoredIndex
    replaced: bool


def index_book(
    path: str | Path,
    *,
    session: Session,
    title: str | None = None,
    settings: Settings | None = None,
    model=None,
    reset: bool = False,
) -> IndexResult:
    """Run Steps 1-4 for one file inside the caller's transaction.

    Args:
        path: Book file to index (.pdf/.txt/.md/.json).
        session: Open SQLAlchemy session. The caller owns commit/rollback;
            this function only flushes.
        title: Optional display title; defaults to the file stem.
        settings: App settings; defaults to ``get_settings()``. Its
            ``chunk_size``, ``chunk_overlap``, and embedding options drive
            Steps 2 and 3.
        model: Optional preloaded/fake encoder, forwarded to
            ``embed_chunks`` (useful in tests and for reusing a warm model).
        reset: When True, clear the existing index first so a different
            embedding model can be used.

    Returns:
        IndexResult with the Step 1-4 outputs and whether a previous copy of
        the same file (matched by sha256) was replaced.

    Raises:
        FileNotFoundError: The path does not exist.
        ValueError: The file cannot be read, produces no chunks, or the
            embedding model/dimension conflicts with the stored index.
    """
    if settings is None:
        settings = get_settings()

    # Step 1: file -> Book
    book = ingest_book(path, title=title)

    # Step 2: Book -> list[Chunk]
    chunks = chunk_book(
        book,
        max_chars=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise ValueError(f"file {path} produced no chunks")

    # Step 3: chunks -> normalized vectors
    embedded = embed_chunks(chunks, settings, model=model)

    # Step 4: write rows in the caller's transaction
    replaced = get_document_id(session, book.sha256) is not None
    if reset:
        reset_index(session)
        replaced = False

    stored = store_index(session, book, chunks, embedded)
    session.flush()

    return IndexResult(
        book=book,
        chunks=chunks,
        embedded=embedded,
        stored=stored,
        replaced=replaced,
    )
