"""Step 1 - Ingest: file -> clean chapters + page metadata.

Where the input comes from:
    The caller (API upload / CLI / pipeline) passes a file path.

What this step produces:
    Book(id, sha256, title, source_path, chapters=[Chapter, ...])
    - Book.id is the file sha256, used as the stable document identity.
    - Each Chapter keeps page_start/page_end so later citations work.

Who consumes it:
    Step 2 (app.rag.step2_chunk.chunk_book) reads Book.chapters.
    Step 4 (app.rag.step4_store.store_index) reads Book metadata to persist rows.

This step does NOT touch the database or embeddings.
"""

from dataclasses import dataclass

from pathlib import Path
from pypdf import PdfReader
import hashlib
import re
import json
@dataclass
class Chapter:
    ordinal:int
    title:str
    text:str
    page_start:int|None
    page_end:int | None


@dataclass
class Book:
    id:str
    sha256:str
    title:str
    source_path:str
    chapters: list[Chapter]


SUPPORTED_SUFFIXES={".pdf",".txt",".md",".json",}
#This accepts Chapter 1, Chapter 1: The Storm, and Chapter IV - Beginnings, but not Chapter 1.1 or Chapter One
_CHAPTER_HEADING = re.compile(
    r"^\s*(?:chapter|part|book)\s+(?:\d+|[ivxlcdm]+)"
    r"\s*(?:[:.\-–—]\s*.*)?$",
    re.IGNORECASE,
)
#Also allow Markdown headings with up to three leading spaces:
_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,3}\s+(.+?)\s*#*\s*$")


# this help in processing the book into chapter text and also help in cleaning the
# text and removing the extra spaces and new lines and also help in calculating the sha256 hash of the file

def ingest_book(path :str| Path ,title:str | None =None) -> Book:
    """Turn one supported file into a Book of cleaned, page-aware chapters.

    Input comes from the caller (API/CLI/pipeline): a path to .pdf/.txt/.md/.json.
    Data flow inside this function:
        1. validate the path and suffix
        2. _sha256_file(path)      -> stable book/document id
        3. _read_pages(path)       -> [(page_number, raw_text), ...]
        4. _clean_text(raw_text)   -> normalized text per page
        5. _split_chapters(pages)  -> list[Chapter]
        6. return Book(...)        -> consumed by Step 2

    Args:
        path: File to ingest.
        title: Optional display title; defaults to the file stem.

    Returns:
        Book with id == sha256 and one or more chapters.

    Raises:
        FileNotFoundError: The path does not exist.
        ValueError: The path is not a file, has an unsupported suffix, or
            contains no extractable text.
    """
    file_path=Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"file {file_path} does not exist")

    if not file_path.is_file():
        raise ValueError(f"file {file_path} is not a file")

    suffix=file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"file {file_path} has unsupported suffix {suffix} supported types are: {supported}")

    sha256 = _sha256_file(file_path)
    pages = _read_pages(file_path)

    chapters=[]

    cleaned_pages=[
            (page_number, _clean_text(text))
            for page_number, text in pages
        ]
    chapters=_split_chapters(cleaned_pages)

    if not chapters:
            raise ValueError(f"file {file_path} does not contain any text")

    return Book(
            id=sha256,
            sha256=sha256,
            title=title or file_path.stem,
            source_path=str(file_path),
            chapters=chapters
        )





def _sha256_file(path:Path) -> str:
    """Hash the whole file, reading 1 MB at a time.

    Called by ingest_book(). The digest becomes Book.id/Book.sha256 and is
    later used by Step 4 to detect re-ingesting the same file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal sha256 digest.
    """
    digest = hashlib.sha256()
    with open(path,"rb") as file:
        while True:
            chunk = file.read(1024*1024)  # read in 1MB chunks
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_pages(path: Path) -> list[tuple[int, str]]:
    """Extract raw text page by page, before cleaning.

    Input comes from ingest_book(path). Output is passed to _clean_text()
    one page at a time.

    Page numbering:
        .pdf       -> real PDF page numbers (1-based)
        .txt/.md   -> one pseudo-page numbered 1
        .json      -> list of {"text": "..."} objects, numbered 1..n

    Args:
        path: Validated input file.

    Returns:
        List of (page_number, raw_text) tuples.
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path)) #first read the if the folder is the pdf

        if reader.is_encrypted:
            raise ValueError(f"Encrypted PDFs are not supported: {path}")

        return [
            (page_number, page.extract_text() or "")
            for page_number, page in enumerate(reader.pages, start=1)
        ]

    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [(1, text)]

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(
                f"json file {path} does not contain a list of pages"
            )

        pages: list[tuple[int, str]] = []

        for page_number, page in enumerate(data, start=1):
            if not isinstance(page, dict):
                raise ValueError(
                    f"json file {path} page {page_number} is not a dict"
                )

            text = page.get("text")

            if not isinstance(text, str):
                raise ValueError(
                    f"json file {path} page {page_number} "
                    "does not contain string 'text'"
                )

            pages.append((page_number, text))

        return pages

    raise ValueError(f"Unsupported suffix: {suffix}")


def _clean_text(text:str)->str:
    r"""Normalize one page of raw extracted text.

    Input comes from _read_pages(); output is fed to _split_chapters().

    Cleaning steps:
1. Normalize line endings
- Convert \r\n and \r to \n.
2. Normalize special spaces
- Replace non-breaking spaces (\u00a0) with normal spaces.
- Remove soft hyphens (\u00ad) if present.
3. Join words split across lines
- A PDF often extracts hyphen-\ .
- Pattern: a word character, hyphen, newline, word character → remove the hyphen and newline.
- Result: hyphenated.
4. Collapse horizontal whitespace
- Multiple spaces and tabs become one space.
- Do not collapse newlines here; paragraph breaks matter.
5. Clean spaces around newlines
- Remove trailing spaces before \n and leading spaces after \n.
6. Limit blank lines
- Three or more consecutive newlines become exactly two.
- This preserves paragraph boundaries without huge empty gaps.
7. Trim the result
- return text.strip().

    Args:
        text (str): _description_

    Returns:
        str: _description_
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u00ad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text) # if a sentance is hyphanated then it will remove the hyphen and join the two words for  eg hyphen-ated to hyphenated
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_chapters(pages: list[tuple[int, str]]) -> list[Chapter]:
    """Group cleaned pages into chapters using heading detection.

    Input comes from ingest_book() after _clean_text() has run on every page.

    Behavior:
        - A detected heading starts a new Chapter.
        - Text before the first heading becomes "Front Matter".
        - If no heading is found anywhere, each non-empty page becomes its
          own chapter named "Page <n>" (fallback for unstructured text).

    Args:
        pages: [(page_number, cleaned_text), ...] in reading order.

    Returns:
        Chapters in book order, each with page_start/page_end filled in.
    """
    chapters: list[Chapter] = []
    current: Chapter | None = None
    last_page: int | None = None
    saw_heading = False

    for page_number, text in pages:
        for line in text.splitlines():
            stripped = line.strip()
            heading = _match_heading(stripped) if stripped else None

            if heading:
                saw_heading = True

                if current is not None and current.text.strip():
                    current.page_end = last_page or current.page_start
                    chapters.append(current)

                current = Chapter(
                    ordinal=len(chapters),
                    title=heading,
                    text="",
                    page_start=page_number,
                    page_end=page_number,
                )
                last_page = page_number
                continue

            if current is None:
                if not stripped:
                    continue
                current = Chapter(
                    ordinal=len(chapters),
                    title="Front Matter",
                    text="",
                    page_start=page_number,
                    page_end=page_number,
                )
                last_page = page_number

            current.text += line + "\n"
            if stripped:
                last_page = page_number

    if current is not None and current.text.strip():
        current.page_end = last_page or current.page_start
        chapters.append(current)

    if not saw_heading:
        return [
            Chapter(
                ordinal=i,
                title=f"Page {page_number}",
                text=text.strip(),
                page_start=page_number,
                page_end=page_number,
            )
            for i, (page_number, text) in enumerate(
                (page for page in pages if page[1].strip())
            )
        ]

    return chapters


def _match_heading(line: str) -> str | None:
    """Return a chapter title if ``line`` looks like a heading.

    Called by _split_chapters() for each non-empty line.

    Matches:
        - Markdown headings: "# Title", "## Title", "### Title"
        - Chapter headings: "Chapter 1", "Chapter IV: The Storm",
          "Part 2 - Beginnings", "Book 3" (max 80 characters)

    Returns:
        The cleaned heading text, or None when the line is body text.
    """
    markdown = _MARKDOWN_HEADING.match(line)
    if markdown:
        return markdown.group(1).strip()

    if len(line) <= 80 and _CHAPTER_HEADING.match(line):
        return line.strip()

    return None
