"""Indexing pipeline: runs Steps 1-4 in order.

Input : a book file path (from the API upload / CLI).
Flow  :
    Step 1 ingest_book()   -> Book (chapters + page metadata)
    Step 2 chunk_book()    -> list[Chunk]
    Step 3 embed_chunks()  -> EmbeddingResult (vectors)
    Step 4 store_index()   -> rows in data/index.db

Step 5 (answer_question) is the read path and is NOT part of this
pipeline; it runs per query against the index written here.

Transaction rule: ``store_index`` only flushes. When this module opens its
own session it commits/rolls back itself; when the caller passes a session,
the caller owns the commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.db.models import Base
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


def _create_engine(settings: Settings) -> Engine:
    """Build the engine for ``settings.database_url``.

    For SQLite the parent directory is created on demand and foreign keys
    are enabled per connection so ``ON DELETE CASCADE`` actually runs.
    """
    url = settings.database_url

    if url.startswith("sqlite"):
        database = make_url(url).database
        if database and database != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)

        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine

    return create_engine(url)


def index_book(
    path: str | Path,
    *,
    title: str | None = None,
    settings: Settings | None = None,
    session: Session | None = None,
    model=None,
    reset: bool = False,
) -> IndexResult:
    """Run Steps 1-4 for one file and persist its index.

    Args:
        path: Book file to index (.pdf/.txt/.md/.json).
        title: Optional display title; defaults to the file stem.
        settings: App settings; defaults to ``get_settings()``. Its
            ``chunk_size``, ``chunk_overlap``, and embedding options drive
            Steps 2 and 3.
        session: Optional SQLAlchemy session. When omitted, this function
            creates the engine, creates missing tables, and commits the
            transaction itself. When provided, the caller owns the commit.
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

    if session is not None:
        return _index_with_session(
            path,
            title=title,
            settings=settings,
            session=session,
            model=model,
            reset=reset,
            commit=False,
        )

    engine = _create_engine(settings)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    with factory() as owned_session:
        try:
            result = _index_with_session(
                path,
                title=title,
                settings=settings,
                session=owned_session,
                model=model,
                reset=reset,
                commit=True,
            )
        except Exception:
            owned_session.rollback()
            raise

    engine.dispose()
    return result


def _index_with_session(
    path: str | Path,
    *,
    title: str | None,
    settings: Settings,
    session: Session,
    model,
    reset: bool,
    commit: bool,
) -> IndexResult:
    """Steps 1-4 against an already-open session.

    ``commit`` is True only when ``index_book`` opened the session itself.
    Keeping the orchestration here avoids duplicating the step calls for the
    owned-session and caller-session paths.
    """
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

    # Step 4: write rows in this transaction
    replaced = get_document_id(session, book.sha256) is not None
    if reset:
        reset_index(session)
        replaced = False

    stored = store_index(session, book, chunks, embedded)

    if commit:
        session.commit()
    else:
        session.flush()

    return IndexResult(
        book=book,
        chunks=chunks,
        embedded=embedded,
        stored=stored,
        replaced=replaced,
    )
