from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from app.rag.step1_ingest import Book


@dataclass
class Chunk:
    """A single chunk of a book.

    Attributes:
        id: Stable chunk identifier (sha256 of book_id + ordinal, truncated).
        book_id: Identifier of the book this chunk belongs to.
        ordinal: Zero-based index of the chunk within the book.
        chapter: Title of the chapter the chunk came from.
        page_start: First page number covered by the chunk, if known.
        page_end: Last page number covered by the chunk, if known.
        text: Chunk text content.
        n_chars: Number of characters in ``text``.
        n_tokens: Estimated token count for ``text``.
    """

    id: str
    book_id: str
    ordinal: int
    chapter: str
    page_start: int | None
    page_end: int | None
    text: str
    n_chars: int
    n_tokens: int


@dataclass
class _Piece:
    """An intermediate text fragment used while packing chunks.

    Attributes:
        text: The fragment text.
        new_paragraph: Whether this fragment starts a new paragraph.
    """

    text: str
    new_paragraph: bool = False


def chunk_book(book: Book, max_chars: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Split a book's chapters into overlapping chunks.

    Validates the size knobs, then processes each chapter independently:
    splits the chapter text into pieces, packs them into chunks of at most
    ``max_chars`` characters, and overlaps adjacent chunks by roughly
    ``overlap`` characters.

    Args:
        book: The book to chunk.
        max_chars: Maximum number of characters allowed in a single chunk.
        overlap: Number of characters of overlap between adjacent chunks.

    Returns:
        A list of ``Chunk`` objects in book order.

    Raises:
        ValueError: If ``max_chars`` is not positive, ``overlap`` is
            negative, or ``overlap`` is not smaller than ``max_chars``.
    """
    # 1. validate the knobs
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    chunks: list[Chunk] = []

    # 2. one chapter at a time
    for chapter in book.chapters:
        if not chapter.text.strip():
            continue  # skip empty chapters

        pieces = _split_pieces(chapter.text, max_chars)
        start = 0

        # 3. pack pieces into chunks until this chapter is used up
        while start < len(pieces):
            end, text = _pack(pieces, start, max_chars)

            ordinal = len(chunks)
            chunks.append(
                Chunk(
                    id=_chunk_id(book.id, ordinal),
                    book_id=book.id,
                    ordinal=ordinal,
                    chapter=chapter.title,
                    page_start=chapter.page_start,
                    page_end=chapter.page_end,
                    text=text,
                    n_chars=len(text),
                    n_tokens=_estimate_tokens(text),
                )
            )

            # 4. all pieces consumed: nothing left to overlap into
            if end >= len(pieces):
                break

            # 5. next chunk starts a little back, to overlap
            start = max(_overlap_start(pieces, end, overlap), start + 1)

    return chunks


# private helpers


def _split_pieces(text: str, max_chars: int) -> list[_Piece]:
    """Split chapter text into intermediate pieces.

    Paragraphs are split on blank lines, then each paragraph is split into
    sentences, and overly long sentences are broken into word groups. Each
    resulting fragment is tagged with whether it begins a new paragraph.

    Args:
        text: The chapter text to split.
        max_chars: Maximum characters allowed in a single piece.

    Returns:
        A list of ``_Piece`` fragments in order.
    """
    pieces: list[_Piece] = []

    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue

        first_piece_of_paragraph = True

        for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue

            parts = (
                [sentence]
                if len(sentence) <= max_chars
                else _split_long_sentence(sentence, max_chars)
            )

            for part in parts:
                pieces.append(
                    _Piece(text=part, new_paragraph=first_piece_of_paragraph)
                )
                first_piece_of_paragraph = False

    return pieces


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Split a sentence that exceeds ``max_chars`` into word groups.

    Words are greedily packed so no group exceeds ``max_chars``. A single
    word longer than ``max_chars`` raises an error.

    Args:
        sentence: The sentence to split.
        max_chars: Maximum characters allowed in a single group.

    Returns:
        A list of text groups, each no longer than ``max_chars``.

    Raises:
        ValueError: If a word is longer than ``max_chars``.
    """
    words = sentence.split()
    if not words:
        return []

    parts: list[str] = []
    current = ""

    for word in words:
        if len(word) > max_chars:
            raise ValueError(
                f"word {word!r} is longer than max_chars={max_chars}"
            )

        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            parts.append(current)
            current = word

    if current:
        parts.append(current)

    return parts


def _join(pieces: list[_Piece]) -> str:
    """Join pieces into a single string.

    Pieces that start a new paragraph are separated by a blank line;
    otherwise they are separated by a single space.

    Args:
        pieces: The pieces to join.

    Returns:
        The joined text.
    """
    if not pieces:
        return ""

    parts = [pieces[0].text]
    for piece in pieces[1:]:
        separator = "\n\n" if piece.new_paragraph else " "
        parts.append(separator + piece.text)

    return "".join(parts)


def _pack(pieces: list[_Piece], start: int, max_chars: int) -> tuple[int, str]:
    """Greedily pack pieces starting at ``start`` into a chunk.

    Pieces are added one at a time until adding the next would exceed
    ``max_chars``.

    Args:
        pieces: All pieces available.
        start: Index of the first piece to pack.
        max_chars: Maximum characters allowed in the resulting chunk.

    Returns:
        A tuple ``(end, text)`` where ``end`` is the exclusive end index
        and ``text`` is the packed chunk text.

    Raises:
        ValueError: If the piece at ``start`` alone exceeds ``max_chars``.
    """
    end = start
    text = ""

    while end < len(pieces):
        candidate = _join(pieces[start : end + 1])
        if len(candidate) > max_chars:
            break
        text = candidate
        end += 1

    if end == start:
        raise ValueError(
            f"piece at index {start} is longer than max_chars={max_chars}"
        )

    return end, text


def _overlap_start(pieces: list[_Piece], end: int, overlap: int) -> int:
    """Find the start index for the next chunk to achieve overlap.

    Walks backwards from ``end`` collecting pieces until adding another
    would exceed ``overlap`` characters.

    Args:
        pieces: All pieces available.
        end: Exclusive end index of the current chunk.
        overlap: Desired overlap in characters.

    Returns:
        The index at which the next chunk should start.
    """
    if end <= 0 or overlap <= 0:
        return end

    start = end - 1
    for index in range(end - 1, -1, -1):
        if len(_join(pieces[index:end])) <= overlap:
            start = index
        else:
            break

    return start


def _chunk_id(book_id: str, ordinal: int) -> str:
    """Generate a stable chunk id.

    Args:
        book_id: Identifier of the book.
        ordinal: Chunk ordinal within the book.

    Returns:
        The first 16 hex characters of the sha256 of ``book_id`` + ``ordinal``.
    """
    return hashlib.sha256(f"{book_id}{ordinal}".encode("utf-8")).hexdigest()[:16]


def _estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text``.

    Uses a rough heuristic of 4 characters per token.

    Args:
        text: The text to estimate.

    Returns:
        The estimated number of tokens.
    """
    return math.ceil(len(text) / 4)
