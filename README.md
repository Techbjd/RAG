# Book RAG — a 5-step pipeline with Python + FastAPI

Upload one book, index it once, ask questions with citations.

This repository is a **scaffold and build guide**. The folders are ready; the
code inside them is yours to write. Every piece maps to one of five steps, and
FastAPI only wraps those steps in HTTP endpoints.

```
 book.pdf / book.epub / book.txt
              │
      ┌───────▼────────┐
      │ 1. INGEST      │  file  ->  clean chapters + page metadata
      └───────┬────────┘
              │  data/processed/<book>.chapters.jsonl
      ┌───────▼────────┐
      │ 2. CHUNK       │  chapters -> overlapping passages
      └───────┬────────┘
              │  data/processed/<book>.chunks.jsonl
      ┌───────▼────────┐
      │ 3. EMBED       │  chunks -> normalized vectors
      └───────┬────────┘
              │  vectors + model name + dimension
      ┌───────▼────────┐
      │ 4. STORE       │  vectors + text -> data/index.db (SQLite)
      └───────┬────────┘
              │
   question   │
      ┌───────▼────────┐
      │ 5. ANSWER      │  retrieve top-k -> grounded answer + citations
      └───────┬────────┘
              ▼
     { answer, sources[] }
```

FastAPI exposes steps 1–4 as `POST /api/books/ingest` (one indexing job) and
step 5 as `POST /api/query`.

---

## 1. The five steps at a glance

| # | Step | Main file | Input | Output | Done when |
|---|------|-----------|-------|--------|-----------|
| 1 | Ingest | `app/rag/step1_ingest.py` | PDF / EPUB / TXT / MD | chapters JSONL | chapter count > 0 and page numbers retained |
| 2 | Chunk | `app/rag/step2_chunk.py` | chapters | chunks JSONL | no chunk exceeds the limit; no text lost |
| 3 | Embed | `app/rag/step3_embed.py` | chunks | vectors + metadata | one vector per chunk, consistent dimension |
| 4 | Store | `app/rag/step4_store.py` | chunks + vectors | `data/index.db` | DB counts match; re-ingest does not duplicate |
| 5 | Answer | `app/rag/step5_answer.py` | question + index | answer + sources | answer cites the correct chapter/page |

`app/rag/pipeline.py` runs steps 1–4 in order; `app/services/` connects that
pipeline to the database and API.

---

## 2. Folder structure

```text
book-rag/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app, lifespan, router registration
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py              # shared dependencies (settings, db session, model)
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── books.py         # POST /api/books/ingest, GET/DELETE /api/books
│   │       ├── query.py         # POST /api/query
│   │       └── health.py        # GET /api/health
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # pydantic-settings; reads .env
│   │   └── logging.py           # one logging setup for the whole app
│   ├── rag/                     # pure Python: no FastAPI imports here
│   │   ├── __init__.py
│   │   ├── step1_ingest.py      # file -> chapters
│   │   ├── step2_chunk.py       # chapters -> chunks
│   │   ├── step3_embed.py       # chunks -> vectors
│   │   ├── step4_store.py       # chunks + vectors -> SQLite
│   │   ├── step5_answer.py      # question -> retrieval + answer
│   │   └── pipeline.py          # runs steps 1-4 for one book
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── book.py              # BookOut, IngestResponse, ...
│   │   └── query.py             # QueryRequest, QueryResponse, Source
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py          # engine/session, schema creation
│   │   └── models.py            # documents, chapters, chunks, embeddings
│   └── services/
│       ├── __init__.py
│       ├── ingest_service.py    # orchestrates steps 1-4, idempotency
│       └── query_service.py     # orchestrates step 5
├── data/
│   ├── raw/                     # put your book here (git-ignored)
│   ├── processed/               # generated chapters/chunks JSONL (git-ignored)
│   └── index/                   # SQLite index (git-ignored)
├── scripts/
│   ├── ingest_book.py           # CLI test for steps 1-4 without FastAPI
│   └── ask.py                   # CLI test for step 5
├── tests/
│   ├── fixtures/                # a tiny public-domain or invented sample book
│   ├── test_ingest.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_api.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

**Layering rule:** `rag/` contains plain functions and dataclasses. It must not
import FastAPI. `services/` may use the database. `api/` only parses HTTP,
calls a service, and returns a schema. This keeps the RAG logic testable from a
script and reusable from a worker later.

Only `.gitkeep` files and empty `__init__.py` files exist right now. Create the
remaining files in the build order in section 9.

---

## 3. Requirements and setup

Python 3.11+ is comfortable (3.10 minimum).

```bash
cd book-rag
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

`requirements.txt` includes:

- **fastapi**, **uvicorn[standard]**, **python-multipart** — API and file uploads
- **pydantic-settings** — typed configuration
- **numpy** — vector search and vector math
- **pypdf** — PDF text extraction
- **sentence-transformers** — local embedding model (first run downloads it)
- optional **openai** — hosted embeddings and generated answers

If `sentence-transformers` is too heavy for your machine, install `fastembed`
instead and swap the two lines in `step3_embed.py`; the rest of the pipeline is
unchanged. You can also use OpenAI embeddings and avoid local torch entirely.

For EPUB, either use the standard library (`zipfile` + `xml.etree`) or add
`ebooklib`.

---

## 4. Configuration

`app/core/config.py` should read environment variables with
`pydantic-settings` and expose one `Settings` object:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "book-rag"
    data_dir: str = "./data"
    database_url: str = "sqlite:///./data/index.db"

    chunk_size: int = 1000
    chunk_overlap: int = 150

    embedding_backend: str = "sentence-transformers"  # or "openai"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch: int = 64

    top_k: int = 5
    hybrid_alpha: float = 0.6      # 0 = BM25 only, 1 = vectors only
    min_score: float = 0.20

    llm_backend: str = "extractive"  # or "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
```

The supplied `.env.example` contains the matching variables. Keep `.env` out of
git.

Rules of thumb:

- `CHUNK_SIZE=1000` characters, `CHUNK_OVERLAP=150` is a good start for prose.
- `TOP_K=5`; increase to 8 if answers feel thin.
- `HYBRID_ALPHA=0.6` leans slightly toward vectors; `0.5` is neutral.
- Freeze one embedding model. Switching models later means re-embedding the
  whole book, because old and new vectors are not comparable.

---

## 5. Step 1 — Ingest the book

**Goal:** one file becomes clean chapters with metadata needed for citations.

Create `app/rag/step1_ingest.py` with roughly this public shape:

```python
@dataclass
class Chapter:
    index: int
    title: str
    text: str
    page_start: int | None
    page_end: int | None

@dataclass
class Book:
    id: str              # e.g. first 12 chars of sha256
    sha256: str
    title: str
    source_path: str
    chapters: list[Chapter]

def ingest_book(path: Path, title: str | None = None) -> Book: ...
```

Implementation notes:

1. Reject unsupported extensions early. Start with `.pdf`, `.txt`, `.md`,
   `.epub`.
2. Compute `sha256` of the file bytes. Use it as the stable document ID so
   re-uploading the same book can be detected.
3. **PDF:** use `pypdf.PdfReader`, extract page by page, and keep the page
   number. Group pages into chapters with a heading regex
   (`^(chapter|part|book)\s+\d+`) or treat each page as a chapter if the PDF has
   no structure. Never throw away page numbers: step 5 needs them for
   citations.
4. **TXT / Markdown:** split on `#` / `##` headings, or on `Chapter N` lines.
5. **EPUB:** read the OPF spine (or use `ebooklib`) so chapters arrive in reading
   order. Strip HTML tags; keep `<h1>/<h2>` as chapter titles.
6. **Clean:** normalize line endings, collapse repeated blank lines, remove
   running page headers/footers, fix words hyphenated across line breaks.
   Drop table-of-contents pages and copyright pages if they contain no prose.
7. Save an intermediate artifact so the next step is debuggable:

```text
data/processed/<book_id>.chapters.jsonl
{"kind":"chapter","index":0,"title":"Chapter One","text":"...","page_start":1,"page_end":12}
```

**Acceptance check:** open the JSONL and verify the first and last chapter look
readable and that `page_start` / `page_end` point at real pages.

---

## 6. Step 2 — Chunk

**Goal:** split chapters into passages small enough to embed, large enough to
carry meaning.

Create `app/rag/step2_chunk.py`:

```python
@dataclass
class Chunk:
    id: str                 # stable: sha256(book_id + ordinal)[:16]
    book_id: str
    ordinal: int
    chapter: str
    page_start: int | None
    page_end: int | None
    text: str
    n_chars: int
    n_tokens: int

def chunk_book(book: Book, max_chars: int = 1000, overlap: int = 150) -> list[Chunk]: ...
```

Recommended algorithm:

1. Split chapter text into paragraphs on blank lines.
2. If a paragraph is still too large, split it into sentences at `.`, `?`, `!`.
3. Greedily pack sentences into a chunk until `max_chars` is reached.
4. Start the next chunk with the last whole sentences that fit inside
   `overlap` characters. Overlap preserves context across boundaries.
5. Never split mid-word. Keep chapter title, page range, and ordinal on every
   chunk.
6. Write `data/processed/<book_id>.chunks.jsonl`.

Why 1000 characters and 150 overlap: roughly 200–250 words per passage, enough
for a precise answer, small enough that the top-5 context stays inside a normal
LLM prompt.

**Acceptance check:** every chunk is non-empty and `len(text) <= max_chars`;
searching the concatenated chunks for the first and last sentence of the book
finds both; adjacent chunks visibly overlap.

---

## 7. Step 3 — Embed

**Goal:** turn each chunk into a vector with a **single, frozen** model.

Create `app/rag/step3_embed.py`:

```python
def embed_chunks(chunks: list[Chunk], settings: Settings) -> EmbeddingResult:
    ...
```

Local default:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
vectors = model.encode(
    [c.text for c in chunks],
    batch_size=64,
    normalize_embeddings=True,   # required for cosine similarity
    show_progress_bar=True,
)
```

Details that save time later:

- **One vector per chunk.** Keep `chunk.id` aligned with row order.
- **Normalize** vectors. Then cosine similarity is just a dot product.
- **Batch** 32–64 texts. CPU is fine for one book; a 300-page book is usually a
  few thousand chunks.
- Store the model name and dimension alongside the vectors. You will check them
  in step 4.
- To improve retrieval, embed `chapter title + "\n" + chunk text`, but store and
  display the plain chunk text.
- Optional: cache embeddings as `.npy` plus a JSON metadata file, or write
  straight into the database. The separate artifact keeps steps inspectable.

**Acceptance check:** number of vectors equals number of chunks; every vector
has the same length (384 for `bge-small`, 1536 for OpenAI small); L2 norm is
approximately 1.0.

---

## 8. Step 4 — Store

**Goal:** persist documents, chunks, and vectors so step 5 can search them.

For one book, SQLite plus NumPy brute-force search is enough and avoids running
a vector database. Use `pgvector` or Qdrant only when you outgrow it (see
section 14).

Suggested schema (`app/db/models.py`):

```sql
CREATE TABLE documents (
    id          TEXT PRIMARY KEY,          -- sha-based book id
    sha256      TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    source_path TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE chapters (
    id          INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    title       TEXT NOT NULL,
    page_start  INTEGER,
    page_end    INTEGER
);

CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    chapter     TEXT NOT NULL,
    page_start  INTEGER,
    page_end    INTEGER,
    text        TEXT NOT NULL,
    n_tokens    INTEGER NOT NULL
);

CREATE TABLE embeddings (
    chunk_id  INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    dim       INTEGER NOT NULL,
    vector    BLOB NOT NULL             -- float32 bytes
);

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL                 -- e.g. embedding_signature
);
```

Implementation details:

- Convert vectors with
  `np.asarray(vector, dtype=np.float32).tobytes()`; load with
  `np.frombuffer(blob, dtype=np.float32)`.
- Write one `meta` row: `embedding_signature = model/dim`. On ingest, compare it
  with the new vectors and **refuse to mix models**. Offer `--reset` to rebuild.
- Make ingestion idempotent: delete the existing document with the same
  `sha256` (cascade removes chunks and embeddings) and insert fresh rows inside
  one transaction.
- Load all vectors once per process into a single `numpy` matrix; search with
  `matrix @ query_vector`. For a single book this is milliseconds.
- Keep the SQLite file at `data/index/` (or `data/index.db`); never commit it.

**Acceptance check:**

```sql
SELECT COUNT(*) FROM chunks;       -- equals chunks written in step 2
SELECT COUNT(*) FROM embeddings;   -- equals chunks
SELECT DISTINCT model, dim FROM embeddings;  -- exactly one row
```

Re-run the ingest and confirm the counts stay the same, not double.

---

## 9. Step 5 — Retrieve and answer

**Goal:** given a question, find the best passages and produce a grounded answer.

Create `app/rag/step5_answer.py`:

```python
@dataclass
class Source:
    chunk_id: int
    chapter: str
    page_start: int | None
    page_end: int | None
    score: float
    text: str

@dataclass
class Answer:
    answer: str
    sources: list[Source]

def answer_question(question: str, top_k: int = 5) -> Answer: ...
```

Retrieval:

1. Embed the question with the **same** embedder used for the book.
2. Compute cosine similarity against all chunk vectors.
3. Optionally compute BM25 over the chunk text and combine:
   `score = alpha * cosine + (1 - alpha) * bm25_normalized`.
   Hybrid search handles both exact names and paraphrased questions.
4. Take `top_k` (default 5). Drop results below `MIN_SCORE`; if everything is
   below it, return “That does not appear in this book.” instead of guessing.
5. Optional: merge adjacent chunks from the same chapter to avoid five
   near-duplicates.

Answering, two modes:

- **`extractive` (default, no API key):** return the best passage plus the
  source list. Honest, fast, and useful for testing retrieval.
- **`openai` (or any OpenAI-compatible endpoint):** send numbered passages and
  ask for an answer with `[1]`-style citations. Prompt template:

```text
You are a careful book assistant. Answer only from the passages below.
Cite every claim with the passage number, like [1]. If the passages do not
contain the answer, say "The book does not say." Never use outside knowledge.

[1] Chapter One (pages 3-8)
<passage text>

[2] Chapter Three (pages 41-47)
<passage text>

Question: ...
```

Return `{answer, sources[]}` where each source includes chapter, page range,
score, and an excerpt. Citations are the difference between a demo and a usable
RAG app.

**Acceptance check:** ask a question whose answer you know is on a specific
page; the top source should be that chapter and page. Ask something unrelated;
the app should refuse rather than invent.

---

## 10. FastAPI layer

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/books/ingest` | upload a book; runs steps 1–4 |
| `GET` | `/api/books` | list indexed books |
| `GET` | `/api/books/{book_id}` | book details and chunk count |
| `DELETE` | `/api/books/{book_id}` | remove a book and its vectors |
| `POST` | `/api/query` | step 5: answer a question |
| `GET` | `/api/health` | process + database status |

### Request / response contracts

`POST /api/books/ingest` accepts `multipart/form-data`:

```bash
curl -X POST http://127.0.0.1:8000/api/books/ingest \
  -F "file=@data/raw/my-book.pdf"
```

```json
{
  "book_id": "8f14e45fceea",
  "title": "my-book",
  "chapters": 24,
  "chunks": 3120,
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "replaced": false,
  "duration_ms": 18420
}
```

`POST /api/query`:

```bash
curl -X POST http://127.0.0.1:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Who built the tide clock?", "top_k": 5}'
```

```json
{
  "answer": "The tide clock was built in 1887 by the astronomer Elian Rook [1].",
  "sources": [
    {
      "chunk_id": 412,
      "chapter": "Chapter Two: The Tide Clock",
      "page_start": 31,
      "page_end": 34,
      "score": 0.81,
      "text": "The tide clock stood in the tower..."
    }
  ],
  "latency_ms": 840
}
```

### Wiring notes

- In `app/main.py`, use a `lifespan` handler to create tables and load the
  embedding model **once**. Loading a model per request is the most common
  performance mistake.
- File uploads: `UploadFile`; validate extension and size; save to
  `data/raw/<sha>.pdf`; then call the ingest service.
- CPU-heavy steps block the event loop. For a single book it is acceptable to
  run them in `fastapi.concurrency.run_in_threadpool`, or use `BackgroundTasks`
  plus `GET /api/books/{id}/status` if you want progress.
- Keep API schemas (`app/schemas/`) separate from the internal RAG dataclasses.
  That lets you change internal fields without breaking clients.
- Return meaningful status codes: 400 unsupported file, 409 book already
  indexed (or replace it), 422 bad query, 500 pipeline failure.
- `GET /api/health` should check the DB, not just return `{"ok": true}`.

Minimal app skeleton:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.embedder = load_embedder()
    yield

app = FastAPI(title="Book RAG", lifespan=lifespan)
app.include_router(books.router, prefix="/api/books", tags=["books"])
app.include_router(query.router, prefix="/api/query", tags=["query"])
app.include_router(health.router, prefix="/api", tags=["health"])
```

---

## 11. Run it

```bash
# development server
uvicorn app.main:app --reload

# optional: test steps 1-4 without the API
python scripts/ingest_book.py data/raw/my-book.pdf

# optional: test step 5 without the API
python scripts/ask.py "Who built the tide clock?"
```

Interactive API docs: <http://127.0.0.1:8000/docs>

Suggested curl flow:

```bash
curl http://127.0.0.1:8000/api/health

curl -X POST http://127.0.0.1:8000/api/books/ingest \
  -F "file=@data/raw/my-book.pdf"

curl -X POST http://127.0.0.1:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What happens at low tide?", "top_k": 5}'
```

---

## 12. Build order (from this scaffold to a working API)

Follow the steps in this order. Each checkpoint is runnable before moving on.

1. **Environment** — create venv, install requirements, copy `.env.example` to
   `.env`.
2. **Config** — write `app/core/config.py`; confirm `Settings()` prints the
   expected values.
3. **Database** — write `app/db/database.py` and `models.py`; create tables.
4. **Step 1** — write `step1_ingest.py`; run it from a scratch script on your
   book; inspect the chapters JSONL.
5. **Step 2** — write `step2_chunk.py`; inspect chunk count, lengths, overlap.
6. **Step 3** — write `step3_embed.py`; assert vector shape and norm.
7. **Step 4** — write `step4_store.py`; run the SQL acceptance checks.
8. **Step 5** — write `step5_answer.py`; test retrieval with `extractive` mode
   before adding any LLM key.
9. **Pipeline** — write `pipeline.py` to run 1–4 and `scripts/ingest_book.py`
   to call it.
10. **Schemas + services** — define request/response models and move the
    orchestration out of scripts into `services/`.
11. **API** — implement routes and `main.py`; start uvicorn; ingest and query
    through `/docs`.
12. **Tests** — add the tests below; run `pytest -q`.
13. **Commit** — initialize git and commit (section 13).

A useful rule: **do not write FastAPI code until the five steps work from a
script.** HTTP is the easy part; retrieval quality is the hard part.

---

## 13. Tests to write

- `tests/test_ingest.py` — a tiny fixture `.txt` produces the expected number of
  chapters and keeps page numbers.
- `tests/test_chunking.py` — chunks respect `max_chars`, contain every sentence,
  and overlap.
- `tests/test_retrieval.py` — index a 3–4 chapter fixture; a question about a
  distinctive fact returns the right chapter; an unrelated question returns no
  confident hit.
- `tests/test_api.py` — use `fastapi.testclient.TestClient` with a temporary
  `BOOKRAG_DATA_DIR`; ingest a fixture, query it, assert `sources` is non-empty
  and the answer contains the expected fact.
- Keep tests offline: use the hashing/`extractive` path or monkeypatch the
  embedder; never call a paid API in CI.

```bash
pytest -q
```

---

## 14. Commit

Generated data (`.env`, `data/raw/*`, `data/processed/*`, `data/index/*`) is
already ignored. `.gitkeep` files keep the empty folders in the repository.

```bash
git init
git add .
git status                     # confirm no book/PDF, no .env, no index.db
git commit -m "chore: scaffold 5-step RAG pipeline with FastAPI"
```

If you prefer one commit per step (recommended while learning):

```bash
git add app/rag/step1_ingest.py tests/test_ingest.py
git commit -m "step 1: ingest books into chapters"

git add app/rag/step2_chunk.py tests/test_chunking.py
git commit -m "step 2: chunk chapters with overlap"

git add app/rag/step3_embed.py
git commit -m "step 3: embed chunks with a frozen model"

git add app/rag/step4_store.py app/db/
git commit -m "step 4: store chunks and vectors in SQLite"

git add app/rag/step5_answer.py tests/test_retrieval.py
git commit -m "step 5: hybrid retrieval with cited answers"

git add app/ scripts/ tests/
git commit -m "feat: expose the pipeline through FastAPI"
```

If your book is public domain or you have the rights, remove the
`data/raw/*` ignore rule before committing it; otherwise keep it local and
document where users should drop their own copy.

---

## 15. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| PDF ingests 0 characters | Scanned image PDF | Run OCR first, e.g. `ocrmypdf in.pdf out.pdf` |
| `dimension mismatch` on query | Index built with another embedding model | Re-embed and re-store, or restore the original model |
| First query takes 30+ seconds | Model download / lazy load | Preload in `lifespan`; warm up with one dummy query |
| Answers cite the wrong chapter | Chunks crossed chapter boundaries or page metadata lost | Keep chapter/page on every chunk; never drop page numbers in step 1 |
| Five near-identical sources | Adjacent chunks overlap | Deduplicate by chapter + nearby page, or use MMR |
| Answers sound invented | LLM allowed outside knowledge | Require citations; enforce `MIN_SCORE`; say “not in the book” otherwise |
| Ingest is slow | Embedding on CPU in the request path | Batch, or move steps 1–4 to a background job |
| Re-upload duplicates chunks | No idempotency key | Delete by `sha256` before inserting |
| SQLite `database is locked` | Concurrent writers | Serialize ingest; one job at a time, or move to Postgres |

---

## 16. When to outgrow this design

The five steps stay the same; only the pieces behind them change:

- **Better retrieval:** add a cross-encoder reranker over the top 20 candidates,
  then keep the top 5.
- **More books:** replace SQLite/NumPy with Postgres + pgvector or Qdrant; add
  metadata filters (`book_id`, chapter, page).
- **Production ingest:** queue jobs (Celery/RQ/Arq), report progress, retry
  failures.
- **Streaming answers:** stream tokens over SSE and send sources first.
- **Evaluation:** build a small question → expected-page test set and measure
  recall@5 before changing chunk size or models.

Keep the five-step boundaries. They make every failure easy to locate: bad text,
bad chunks, bad vectors, bad storage, or bad prompting.
