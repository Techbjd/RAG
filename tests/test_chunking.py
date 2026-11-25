import hashlib
import math

import pytest

from app.rag.step1_ingest import Book, Chapter
from app.rag.step2_chunk import chunk_book


def _book(texts, book_id="book-1"):
    return Book(
        id=book_id,
        sha256=book_id,
        title="Test Book",
        source_path="test.txt",
        chapters=[
            Chapter(
                ordinal=i,
                title=f"Chapter {i + 1}",
                text=text,
                page_start=i + 1,
                page_end=i + 1,
            )
            for i, text in enumerate(texts)
        ],
    )


def test_chunk_book_keeps_metadata_and_respects_limits():
    book = _book(["Sentence one. Sentence two. Sentence three. Sentence four."])

    chunks = chunk_book(book, max_chars=25, overlap=10)

    assert len(chunks) == 4
    assert [chunk.ordinal for chunk in chunks] == [0, 1, 2, 3]
    assert [chunk.chapter for chunk in chunks] == ["Chapter 1"] * 4
    assert [(chunk.page_start, chunk.page_end) for chunk in chunks] == [(1, 1)] * 4
    assert all(chunk.book_id == book.id for chunk in chunks)
    assert all(0 < len(chunk.text) <= 25 for chunk in chunks)
    assert all(chunk.n_chars == len(chunk.text) for chunk in chunks)
    assert all(chunk.n_tokens == math.ceil(chunk.n_chars / 4) for chunk in chunks)
    assert len({chunk.id for chunk in chunks}) == 4


def test_chunk_book_ids_are_stable_and_book_specific():
    book = _book(["Sentence one. Sentence two."])
    options = {"max_chars": 15, "overlap": 5}

    first_run = chunk_book(book, **options)
    second_run = chunk_book(book, **options)
    other_book = _book(["Sentence one. Sentence two."], book_id="other-book")

    assert [chunk.id for chunk in first_run] == [chunk.id for chunk in second_run]
    expected = hashlib.sha256(f"{book.id}0".encode("utf-8")).hexdigest()[:16]
    assert first_run[0].id == expected
    assert first_run[0].id != chunk_book(other_book, **options)[0].id


def test_chunk_book_overlaps_adjacent_chunks():
    book = _book([
        "Alpha sentence one. Beta sentence two. "
        "Gamma sentence three. Delta sentence four."
    ])

    chunks = chunk_book(book, max_chars=45, overlap=22)

    assert len(chunks) == 3
    assert "Beta sentence two." in chunks[0].text
    assert "Beta sentence two." in chunks[1].text
    assert "Gamma sentence three." in chunks[1].text
    assert "Gamma sentence three." in chunks[2].text


def test_chunk_book_keeps_all_sentences():
    sentences = [
        "The first sentence introduces the hero.",
        "The hero carries a lantern.",
        "A storm arrives at midnight.",
        "The second paragraph begins here.",
        "The lantern survives the rain.",
        "The final sentence closes the tale.",
    ]
    book = _book([
        "\n\n".join([" ".join(sentences[:3]), " ".join(sentences[3:])])
    ])

    chunks = chunk_book(book, max_chars=80, overlap=30)
    combined = " ".join(chunk.text for chunk in chunks)

    assert sentences[0] in combined
    assert sentences[-1] in combined
    assert all(sentence in combined for sentence in sentences)
    assert any("\n\n" in chunk.text for chunk in chunks)


def test_chunk_book_never_splits_words():
    words = [f"word{i:02d}" for i in range(30)]
    book = _book([" ".join(words)])

    chunks = chunk_book(book, max_chars=20, overlap=5)

    assert all(len(chunk.text) <= 20 for chunk in chunks)
    for chunk in chunks:
        assert set(chunk.text.split()) <= set(words)
    assert all(word in " ".join(c.text for c in chunks) for word in words)


def test_chunk_book_raises_for_word_longer_than_max_chars():
    book = _book(["x" * 50])

    with pytest.raises(ValueError, match="longer than max_chars"):
        chunk_book(book, max_chars=20, overlap=5)


def test_chunk_book_skips_empty_chapters():
    book = _book(["", "Only sentence."])

    chunks = chunk_book(book, max_chars=100, overlap=20)

    assert len(chunks) == 1
    assert chunks[0].chapter == "Chapter 2"
    assert (chunks[0].page_start, chunks[0].page_end) == (2, 2)
    assert chunks[0].text == "Only sentence."


def test_chunk_book_returns_empty_list_for_book_without_chapters():
    assert chunk_book(_book([])) == []


@pytest.mark.parametrize(
    ("max_chars", "overlap"),
    [(0, 0), (-5, 1), (10, -1), (10, 10), (10, 20)],
)
def test_chunk_book_rejects_invalid_sizes(max_chars, overlap):
    with pytest.raises(ValueError):
        chunk_book(_book(["Some text."]), max_chars=max_chars, overlap=overlap)
