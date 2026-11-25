"""Step 4 - Store: vectors + text -> SQLite index.

Input comes from:
    Step 1: Book (document metadata + sha256)
    Step 2: list[Chunk] (text + citation metadata)
    Step 3: EmbeddingResult (model, dim, row-aligned vectors)

Output:
    Rows in data/index.db: Document/Chapter/Chunk/Embedding/Meta.
    On read, load_index() returns StoredIndex(document_id, model, dim,
    chunks, vectors).

Who consumes it:
    Step 5 (app.rag.step5_answer.answer_question) calls load_index() for
    every stored document.

Transaction rule: these helpers flush but never commit; the caller owns
the transaction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session


from app.db.models import Chapter as ChapterRow
from app.db.models import Chunk as ChunkRow
from app.db.models import Document, Embedding, Meta
from app.rag.step1_ingest import Book
from app.rag.step2_chunk import Chunk
from app.rag.step3_embed import EmbeddingResult


META_SIGNATURE_KEY = "embedding_signature"


@dataclass
class StoredIndex:
    """

    Attributes:
        document_id: Stored document ID.
        model: Frozen embedding model name.
        dim: Embedding dimension.
        chunks: Chunks in stored ordinal order.
        vectors: Row-aligned vector matrix.
    """

    document_id: str
    model: str
    dim: int
    chunks: list[Chunk]
    vectors: np.ndarray


def embedding_signature(model: str, dim: int) -> str:
    """Return the model/dimension fingerprint written to ``meta``."""
    return f"{model}/{dim}"






def validate_embedding_result(
    chunks: list[Chunk],
    embedded: EmbeddingResult,
) -> None:
    """Ensure vectors are finite and row-aligned with ``chunks``."""
    ...


def _stored_signature(session: Session) -> str | None:
    """Read the current ``meta.embedding_signature`` value, if any."""
    row = session.get(Meta,META_SIGNATURE_KEY)
    return row.value if row is not None else None
    ...

def signatures_match(stored: str | None, current: str) -> bool:
    """True when there is no stored signature yet or it equals ``current``.

    Called by _ensure_signature_compatible() during store_index().
    A mismatch means the embedding model/dimension changed and the index
    must be reset before new vectors can be written.
    """
    return stored is None or stored == current

def _ensure_signature_compatible(session: Session, signature: str) -> None:
    """Refuse to write when the stored model/dimension differs."""
    stored=_stored_signature(session)

    if not signatures_match(stored,signature):
        raise ValueError(
            "Embedding model changed .Run reset_index first"
        )



def _set_stored_signature(session: Session, signature: str) -> None:
    """Insert or update ``meta.embedding_signature``."""
    row =session.get(Meta,META_SIGNATURE_KEY)

    if row is None:
        session.add(Meta(key=META_SIGNATURE_KEY, value=signature))
    else:
        row.value=signature

    session.flush()


def reset_index(session: Session) -> None:
    """Delete all indexed rows so a new embedding model can be used.

    Clears the stored embedding signature; the caller then re-runs Steps 1-4.
    """
    row =session.get(Meta,META_SIGNATURE_KEY)
    if row is not None:
        session.delete(row)
    session.flush()



def delete_document(session: Session, document_id: str) -> None:
    """Delete one document; cascades remove its chapters/chunks/embeddings.

    Called by store_index() when the same sha256 is re-ingested, and by the
    API when a book is removed.
    """
    row=session.get(Document,document_id)
    if row is not None:
        session.delete(row)

    session.flush()


def get_document_id(session: Session, sha256: str) -> str | None:
    """Look up a stored document by its file hash."""
    return session.scalar(# gives the first column value
        select(Document.id).where(Document.sha256 == sha256)
    )



def store_index(
    session: Session,
    book: Book,
    chunks: list[Chunk],
    embedded: EmbeddingResult,
) -> StoredIndex:
    """Write document, chapters, chunks, and vectors in the caller's transaction.

    Input comes from Steps 1-3 (book, chunks, embedded).
    Re-ingesting the same ``sha256`` should replace the old document instead
    of creating duplicate rows. This function flushes but does not commit;
    transaction control belongs to the caller.
    """

    signature= embedding_signature(embedded.model,embedded.dim)
    validate_embedding_result(chunks,embedded)
    _ensure_signature_compatible(session,signature)
    #if shame sha256 delete all
    old_id = get_document_id(session, book.sha256)

    if old_id is not None:
        delete_document(session, old_id)

    session.flush()
    _set_stored_signature(session,signature)


def load_index(session: Session, document_id: str) -> StoredIndex | None:
    """Load chunks and vectors in ordinal order for Step 5.

    Input: ``document_id`` comes from the Document table (Step 5 iterates
    every stored document). The rows were written by store_index().
    Returns ``None`` when the document does not exist. Chunk IDs come from
    the stable ``uid`` column so retrieval sources match Step 2 IDs.
    """
    rows = session.execute(
        select(
            ChunkRow,
            Embedding.embedding,
            Embedding.model,
            Embedding.dim,
        )
        .join(Embedding, Embedding.chunk_id == ChunkRow.id)
        .where(ChunkRow.document_id == document_id)
        .order_by(ChunkRow.ordinal)
    ).all()

    if not rows:
        return None

    chunks = [
        Chunk(
            id=chunk_row.uid,
            book_id=document_id,
            ordinal=chunk_row.ordinal,
            chapter=chunk_row.chapter_title,
            page_start=chunk_row.page_start,
            page_end=chunk_row.page_end,
            text=chunk_row.text,
            n_chars=chunk_row.n_chars,
            n_tokens=chunk_row.n_tokens,
        )
        for chunk_row, vector, model, dim in rows
    ]

    vectors = np.asarray(
        [vector for chunk_row, vector, model, dim in rows],
        dtype=np.float32,
    )

    model = rows[0][2]
    dim = rows[0][3]

    return StoredIndex(
        document_id=document_id,
        model=model,
        dim=dim,
        chunks=chunks,
        vectors=vectors,
    )
