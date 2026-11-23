from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from app.rag.step1_ingest import Book


@dataclass
class Chunk:
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
    text: str
    new_paragraph: bool = False


def chunk_book(book: Book, max_chars: int = 1000, overlap: int = 150) -> list[Chunk]:
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
    if not pieces:
        return ""

    parts = [pieces[0].text]
    for piece in pieces[1:]:
        separator = "\n\n" if piece.new_paragraph else " "
        parts.append(separator + piece.text)

    return "".join(parts)


def _pack(pieces: list[_Piece], start: int, max_chars: int) -> tuple[int, str]:
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
    return hashlib.sha256(f"{book_id}{ordinal}".encode("utf-8")).hexdigest()[:16]


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)
