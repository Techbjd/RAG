import hashlib
import json

import pytest

from app.rag import step1_ingest as ingest


def _write_text(path, content):
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# ingest_book: public API
# ---------------------------------------------------------------------------

def test_ingest_txt_splits_chapters_and_keeps_metadata(tmp_path):
    path = _write_text(
        tmp_path / "my_book.txt",
        "Chapter 1: The Beginning\nOnce upon a time.\n\n"
        "Chapter 2 - The Storm\nThe rain fell.\n",
    )

    book = ingest.ingest_book(path)

    assert isinstance(book, ingest.Book)
    expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert book.sha256 == expected_hash
    assert book.id == expected_hash
    assert book.title == "my_book"
    assert book.source_path == str(path)

    assert [chapter.ordinal for chapter in book.chapters] == [0, 1]
    assert [chapter.title for chapter in book.chapters] == [
        "Chapter 1: The Beginning",
        "Chapter 2 - The Storm",
    ]
    assert all(chapter.page_start == 1 for chapter in book.chapters)
    assert all(chapter.page_end == 1 for chapter in book.chapters)
    assert "Once upon a time." in book.chapters[0].text
    assert "The rain fell." in book.chapters[1].text


def test_ingest_book_uses_explicit_title(tmp_path):
    path = _write_text(tmp_path / "file.txt", "Chapter 1\nSome text.\n")

    book = ingest.ingest_book(path, title="A Better Title")

    assert book.title == "A Better Title"


def test_ingest_markdown_headings(tmp_path):
    path = _write_text(
        tmp_path / "notes.md",
        "# Introduction\nWelcome.\n\n## Deep Dive\nDetails here.\n",
    )

    book = ingest.ingest_book(path)

    assert [chapter.title for chapter in book.chapters] == [
        "Introduction",
        "Deep Dive",
    ]
    assert [chapter.page_start for chapter in book.chapters] == [1, 1]


def test_ingest_json_pages(tmp_path):
    path = _write_text(
        tmp_path / "pages.json",
        json.dumps([
            {"page": 1, "text": "Chapter 1\nAlpha text."},
            {"page": 2, "text": "Chapter 2\nBeta text."},
        ]),
    )

    book = ingest.ingest_book(path)

    assert [chapter.title for chapter in book.chapters] == [
        "Chapter 1",
        "Chapter 2",
    ]
    assert [chapter.page_start for chapter in book.chapters] == [1, 2]
    assert [chapter.page_end for chapter in book.chapters] == [1, 2]
    assert "Alpha text." in book.chapters[0].text
    assert "Beta text." in book.chapters[1].text


def test_ingest_pdf_pages_and_empty_extraction(monkeypatch, tmp_path):
    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        is_encrypted = False

        def __init__(self, path):
            assert path == str(pdf_path)
            self.pages = [FakePage("Chapter 1\nFrom the PDF."), FakePage(None)]

    monkeypatch.setattr(ingest, "PdfReader", FakeReader)
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf")

    book = ingest.ingest_book(pdf_path)

    assert len(book.chapters) == 1
    assert book.chapters[0].title == "Chapter 1"
    assert book.chapters[0].page_start == 1
    assert book.chapters[0].page_end == 1
    assert "From the PDF." in book.chapters[0].text


def test_ingest_book_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ingest.ingest_book(tmp_path / "missing.txt")


def test_ingest_book_rejects_directory(tmp_path):
    directory = tmp_path / "book.txt"
    directory.mkdir()

    with pytest.raises(ValueError, match="is not a file"):
        ingest.ingest_book(directory)


def test_ingest_book_rejects_unsupported_suffix(tmp_path):
    path = _write_text(tmp_path / "book.epub", "Chapter 1\nText.")

    with pytest.raises(ValueError, match="unsupported suffix"):
        ingest.ingest_book(path)


def test_ingest_book_rejects_file_without_text(tmp_path):
    path = _write_text(tmp_path / "empty.txt", "")

    with pytest.raises(ValueError, match="does not contain any text"):
        ingest.ingest_book(path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_sha256_file_hashes_full_content(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"abc")

    assert ingest._sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_read_pages_pdf_rejects_encrypted_file(monkeypatch, tmp_path):
    class FakeReader:
        is_encrypted = True

        def __init__(self, path):
            pass

    monkeypatch.setattr(ingest, "PdfReader", FakeReader)
    pdf_path = tmp_path / "locked.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf")

    with pytest.raises(ValueError, match="Encrypted PDFs are not supported"):
        ingest._read_pages(pdf_path)


@pytest.mark.parametrize(
    "data",
    [
        {"text": "not a list"},
        ["not a dict"],
        [{}],
        [{"text": 123}],
    ],
)
def test_read_pages_json_validation(tmp_path, data):
    path = _write_text(tmp_path / "pages.json", json.dumps(data))

    with pytest.raises(ValueError, match="json file"):
        ingest._read_pages(path)


def test_clean_text_normalizes_line_endings_and_whitespace():
    raw = (
        "Line 1\r\n"
        "A\u00a0nonbreaking and soft\u00adly space\r\n"
        " hyphen-\nated   with   spaces  \n\n\n\n"
        "  Last line  "
    )

    assert ingest._clean_text(raw) == (
        "Line 1\n"
        "A nonbreaking and softly space\n"
        "hyphenated with spaces\n\n"
        "Last line"
    )


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Chapter 1", "Chapter 1"),
        ("Chapter IV - Beginnings", "Chapter IV - Beginnings"),
        ("PART 2: The Journey", "PART 2: The Journey"),
        ("Book 3", "Book 3"),
        ("   # Introduction", "Introduction"),
        ("### Deep Dive ###", "Deep Dive"),
        ("Chapter One", None),
        ("#NoSpace", None),
        ("    #### Too Deep", None),
        ("Just a sentence.", None),
        ("Chapter 1: " + "x" * 80, None),
    ],
)
def test_match_heading(line, expected):
    assert ingest._match_heading(line) == expected


def test_split_chapters_without_headings_returns_one_chapter_per_page():
    pages = [(1, "First page text"), (2, ""), (3, "Third page text")]

    chapters = ingest._split_chapters(pages)

    assert [chapter.title for chapter in chapters] == ["Page 1", "Page 3"]
    assert [chapter.ordinal for chapter in chapters] == [0, 1]
    assert [chapter.page_start for chapter in chapters] == [1, 3]
    assert [chapter.page_end for chapter in chapters] == [1, 3]
    assert [chapter.text for chapter in chapters] == [
        "First page text",
        "Third page text",
    ]


def test_split_chapters_handles_front_matter_and_page_ranges():
    pages = [
        (1, "Preface text."),
        (2, "Chapter 1\nStory starts here."),
        (3, "The story continues."),
        (4, "Chapter 2\nSecond story."),
    ]

    chapters = ingest._split_chapters(pages)

    assert [chapter.title for chapter in chapters] == [
        "Front Matter",
        "Chapter 1",
        "Chapter 2",
    ]
    assert [chapter.ordinal for chapter in chapters] == [0, 1, 2]
    assert [(c.page_start, c.page_end) for c in chapters] == [
        (1, 1),
        (2, 3),
        (4, 4),
    ]
    assert chapters[1].text == "Story starts here.\nThe story continues.\n"
    assert chapters[2].text == "Second story.\n"


def test_split_chapters_ignores_blank_pages_when_tracking_page_end():
    pages = [(1, "Chapter 1\nBody text."), (2, "   \n"), (3, "")]

    chapters = ingest._split_chapters(pages)

    assert len(chapters) == 1
    assert chapters[0].page_start == 1
    assert chapters[0].page_end == 1
