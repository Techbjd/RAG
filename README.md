# Book RAG — a 5-step pipeline with Python + FastAPI

Upload one book, index it once, ask questions with citations.

This repository is a **scaffold and build guide**. The folders are ready; the code inside them is yours to write. Every piece maps to one of five steps, and FastAPI only wraps those steps in HTTP endpoints.

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
      │ 3. EMBED       │  chunks -> MiniLM vectors (384-dim, normalized)
      └───────┬────────┘
              │  vectors + model name + dimension
      ┌───────▼────────┐
      │ 4. STORE       │  vectors + text -> data/index.db (SQLite)
      └───────┬────────┘
              │
   question   │
      ┌───────▼────────┐
      │ 5. ANSWER      │  hybrid search -> rerank -> grounded answer
      └───────┬────────┘
              ▼
     { answer, sources[] }
```

FastAPI exposes steps 1–4 as `POST /api/books/ingest` (one indexing job) and step 5 as `POST /api/query`.

---

## 1. The five steps at a glance

| \# | Step | Main file | Input | Output | Done when |
| --- | --- | --- | --- | --- | --- |
| 1 | Ingest | `app/rag/step1_ingest.py` | PDF / EPUB / TXT / MD | chapters JSONL | chapter count &gt; 0 and page numbers retained |
| 2 | Chunk | `app/rag/step2_chunk.py` | chapters | chunks JSONL | no chunk exceeds the limit; no text lost |
| 3 | Embed | `app/rag/step3_embed.py` | chunks | vectors + metadata | one vector per chunk, consistent dimension |
| 4 | Store | `app/rag/step4_store.py` | chunks + vectors | `data/index.db` | DB counts match; re-ingest does not duplicate |
| 5 | Answer | `app/rag/step5_answer.py` | question + index | answer + sources | answer cites the correct chapter/page |

`app/rag/pipeline.py` runs steps 1–4 in order; `app/services/` connects that pipeline to the database and API.

### Recommended stack (MiniLM + hybrid + rerank)

This scaffold defaults to a small, fast, fully local stack. It is a good fit for a single book on a laptop CPU and still scales to a hosted LLM for answers.

| Layer | Choice | Why |
| --- | --- | --- |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, no query prefix, fast on CPU, ~80 MB |
| Keyword search | BM25 via `rank-bm25` | Catches exact names and rare terms that vectors miss |
| Hybrid score | `alpha * cosine + (1 - alpha) * bm25_normalized` | Combines semantic and keyword recall |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Jointly scores (query, passage); much more precise than the embedder alone |
| Answer model | Gemini (or OpenAI-compatible, or `extractive`) | Only the final top chunks are sent; grounded + cited |
| Vector store | SQLite + NumPy (current) | Enough for one book; move to pgvector/Qdrant at scale |

### Token limits that matter

Three different limits are easy to confuse. Only the first two constrain chunk size; the LLM window is a budget, not a target.

| Component | Max input | Practical limit |
| --- | --- | --- |
| `all-MiniLM-L6-v2` embedder | 256 tokens | ~700–800 characters per chunk (1000 chars can silently truncate) |
| `ms-marco-MiniLM-L-6-v2` reranker | ~512 tokens | query + passage must fit together |
| Gemini / LLM | very large | send only the final top 5 chunks (~2–3K tokens) |

**Important distinctions:**

- **Embedding ≠ reranking.** The embedder turns text into vectors once at ingest. The reranker is a *separate* cross-encoder that reorders the retrieved candidates at query time. Do **not** “rerank” by re-embedding candidates with the same embedder — that usually changes little.
- **MiniLM needs no prefix.** Unlike BGE, `all-MiniLM-L6-v2` does not require `"Represent this sentence..."` in front of queries. Any question can be encoded as-is.
- **Dimension is frozen.** MiniLM and BGE-small are both 384-dim, so the SQLite schema, BLOB storage, and NumPy search stay unchanged. Switching to a different-dimension model means re-embedding the whole book.
- **Context is a budget, not a target.** A large LLM window does not mean you should fill it. Send the final `top_k` chunks, reserve room for the system prompt, history, and answer, and skip the LLM when nothing clears `MIN_SCORE`.
- **Watch “lost in the middle”.** Models attend better to the beginning and end of a long context. Keep the strongest chunks first/last and avoid padding the prompt with near-duplicates.
- **No folder restructuring is required.** The five-step layout still works. The new retrieval stages can live inside `step5_answer.py`, or be split into optional `retrieval.py` / `rerank.py` files if you prefer smaller modules (see the folder tree below).

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
│   │   ├── step3_embed.py       # chunks -> MiniLM vectors (384-dim)
│   │   ├── step4_store.py       # chunks + vectors -> SQLite
│   │   ├── step5_answer.py      # hybrid retrieval + rerank + answer
│   │   ├── retrieval.py         # OPTIONAL: vector + BM25 hybrid search
│   │   ├── rerank.py            # OPTIONAL: cross-encoder reranker
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
│   ├── test_rerank.py           # OPTIONAL: cross-encoder ordering
│   └── test_api.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

**Layering rule:** `rag/` contains plain functions and dataclasses. It must not import FastAPI. `services/` may use the database. `api/` only parses HTTP, calls a service, and returns a schema. This keeps the RAG logic testable from a script and reusable from a worker later.

**Do you need to change the folder structure?** No — the five-step layout is still correct. `retrieval.py` and `rerank.py` are optional conveniences; if you prefer fewer files, keep both stages as functions inside `step5_answer.py`. The database schema and the `Chunk` / `EmbeddingResult` dataclasses do **not** change, because the default MiniLM embedder is still 384-dimensional.

Only `.gitkeep` files and empty `__init__.py` files exist right now. Create the remaining files in the build order in section 9.

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

- **fastapi**, **uvicorn\[standard\]**, **python-multipart** — API and file uploads
- **pydantic-settings** — typed configuration
- **numpy** — vector search and vector math
- **pypdf** — PDF text extraction
- **sentence-transformers** — local embedding model and cross-encoder reranker (first run downloads them)
- **rank-bm25** — keyword half of hybrid search
- optional **google-genai** or **openai** — hosted answer generation (Gemini / OpenAI-compatible)

The default local models are `sentence-transformers/all-MiniLM-L6-v2` (embedder, 384-dim, no query prefix) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker). Both are small enough for a laptop CPU.

If `sentence-transformers` is too heavy for your machine, install `fastembed` instead and swap the two lines in `step3_embed.py`; the rest of the pipeline is unchanged. You can also use a hosted embedding API and avoid local torch entirely — keep one model and one dimension either way.

For EPUB, either use the standard library (`zipfile` + `xml.etree`) or add `ebooklib`.

---

## 4. Configuration

`app/core/config.py` should read environment variables with `pydantic-settings` and expose one `Settings` object:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "book-rag"
    data_dir: str = "./data"
    database_url: str = "sqlite:///./data/index.db"

    chunk_size: int = 1000
    chunk_overlap: int = 150

    # Step 3: embeddings
    embedding_backend: str = "sentence-transformers"  # or "openai"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, no prefix
    embedding_dim: int = 384
    embedding_batch: int = 64

    # Step 5: retrieval + rerank
    candidate_k: int = 30          # retrieve wide before reranking
    top_k: int = 5                 # final chunks sent to the LLM
    hybrid_alpha: float = 0.6      # 0 = BM25 only, 1 = vectors only
    min_score: float = 0.20
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Step 5: answer generation
    llm_backend: str = "extractive"  # or "gemini" / "openai"
    llm_model: str = "gemini-2.0-flash"
    gemini_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
```

The supplied `.env.example` contains the matching variables. Keep `.env` out of git.

Rules of thumb:

- `CHUNK_SIZE=1000` characters, `CHUNK_OVERLAP=150` is a good start for prose.
- `CANDIDATE_K=30`, then rerank to `TOP_K=5`. Retrieving only 5 and reranking them leaves too little room for the reranker to fix a miss.
- `HYBRID_ALPHA=0.6` leans slightly toward vectors; `0.5` is neutral. Normalize BM25 to `[0, 1]` before mixing, otherwise its raw scores dominate.
- `all-MiniLM-L6-v2` needs **no query prefix** — encode the user's question exactly as typed. BGE models would ask for a fixed instruction prefix; MiniLM does not.
- `EMBEDDING_DIM=384` is part of the storage contract. SQLite BLOBs and the NumPy search matrix assume it. Changing dim means re-embedding and re-storing the whole book.
- `CHUNK_SIZE=700–800` is safer for MiniLM’s 256-token limit. The scaffold default is 1000 characters; long or dense chunks can be silently truncated, so lower it or count tokens with the model tokenizer.
- `TOP_K=5` is the **LLM budget**; `CANDIDATE_K=30` is the **reranker budget**. Do not send `candidate_k` chunks to the answer model.
- Reserve tokens explicitly: system prompt (~300–500), question (~50–200), context (~1.5–3K), history (~500–1K), answer (~500–1K). A ~4–6K-token request is normal even with a 1M-token window.
- Log token counts per chunk and per query. A silent truncation warning is cheaper than debugging bad retrieval later.
- Freeze one embedding model. Switching models later means re-embedding the whole book, because old and new vectors are not comparable. The reranker is separate and can be swapped freely.
- Freeze one reranker model per deployment if reproducibility matters; its scores are only used for ordering candidates, not stored in the database.

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

1. Reject unsupported extensions early. Start with `.pdf`, `.txt`, `.md`, `.epub`.
2. Compute `sha256` of the file bytes. Use it as the stable document ID so re-uploading the same book can be detected.
3. **PDF:** use `pypdf.PdfReader`, extract page by page, and keep the page number. Group pages into chapters with a heading regex (`^(chapter|part|book)\s+\d+`) or treat each page as a chapter if the PDF has no structure. Never throw away page numbers: step 5 needs them for citations.
4. **TXT / Markdown:** split on `#` / `##` headings, or on `Chapter N` lines.
5. **EPUB:** read the OPF spine (or use `ebooklib`) so chapters arrive in reading order. Strip HTML tags; keep `<h1>/<h2>` as chapter titles.
6. **Clean:** normalize line endings, collapse repeated blank lines, remove running page headers/footers, fix words hyphenated across line breaks. Drop table-of-contents pages and copyright pages if they contain no prose.
7. Save an intermediate artifact so the next step is debuggable:

```text
data/processed/<book_id>.chapters.jsonl
{"kind":"chapter","index":0,"title":"Chapter One","text":"...","page_start":1,"page_end":12}
```

**Acceptance check:** open the JSONL and verify the first and last chapter look readable and that `page_start` / `page_end` point at real pages.

---

## 6. Step 2 — Chunk

**Goal:** split chapters into passages small enough to embed, large enough to carry meaning.

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
4. Start the next chunk with the last whole sentences that fit inside `overlap` characters. Overlap preserves context across boundaries.
5. Never split mid-word. Keep chapter title, page range, and ordinal on every chunk.
6. Write `data/processed/<book_id>.chunks.jsonl`.

Why 1000 characters and 150 overlap: roughly 150–200 words per passage, enough for a precise answer, small enough that the top-5 context stays inside a normal LLM prompt.

**MiniLM truncation warning:** `all-MiniLM-L6-v2` accepts only 256 tokens. A 1000-character chunk is often ~200–300 tokens depending on density, so some chunks will be silently truncated before embedding. Prefer **700–800 characters** with MiniLM (or use a 512-token embedder such as BGE-small if you need longer chunks). If you keep 1000, count tokens with the model tokenizer and split on the token limit rather than assuming characters map cleanly to tokens.

**Acceptance check:** every chunk is non-empty and `len(text) <= max_chars`; searching the concatenated chunks for the first and last sentence of the book finds both; adjacent chunks visibly overlap.

---

## 7. Step 3 — Embed

**Goal:** turn each chunk into a vector with a **single, frozen** model.

Create `app/rag/step3_embed.py`:

```python
def embed_chunks(chunks: list[Chunk], settings: Settings) -> EmbeddingResult:
    ...
```

Local default (MiniLM):

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
vectors = model.encode(
    [c.text for c in chunks],
    batch_size=64,
    normalize_embeddings=True,   # required for cosine similarity
    show_progress_bar=True,
)
```

There is no query prefix with MiniLM. At query time, encode the user's question exactly as typed:

```python
query_vector = model.encode(question, normalize_embeddings=True)
```

Details that save time later:

- **One vector per chunk.** Keep `chunk.id` aligned with row order.
- **Normalize** vectors. Then cosine similarity is just a dot product.
- **Batch** 32–64 texts. CPU is fine for one book; a 300-page book is usually a few thousand chunks.
- Store the model name and dimension alongside the vectors. You will check them in step 4.
- To improve retrieval, embed `chapter title + "\n" + chunk text`, but store and display the plain chunk text.
- Optional: cache embeddings as `.npy` plus a JSON metadata file, or write straight into the database. The separate artifact keeps steps inspectable.
- **MiniLM truncates silently past 256 tokens.** Keep chunks ≤ ~700–800 characters, or count tokens with `model.tokenizer` and split on token limits. Truncated embeddings produce vectors that look valid but lose the tail of the passage.
- **Do not swap models after indexing.** `all-MiniLM-L6-v2` is 384-dim; a different model with a different dimension cannot share the stored matrix.

**Acceptance check:** number of vectors equals number of chunks; every vector has length 384; L2 norm is approximately 1.0.

---

## 8. Step 4 — Store

**Goal:** persist documents, chunks, and vectors so step 5 can search them.

For one book, SQLite plus NumPy brute-force search is enough and avoids running a vector database. Use `pgvector` or Qdrant only when you outgrow it (see section 14).

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

- Convert vectors with `np.asarray(vector, dtype=np.float32).tobytes()`; load with `np.frombuffer(blob, dtype=np.float32)`.
- Write one `meta` row: `embedding_signature = model/dim`. On ingest, compare it with the new vectors and **refuse to mix models**. Offer `--reset` to rebuild.
- Make ingestion idempotent: delete the existing document with the same `sha256` (cascade removes chunks and embeddings) and insert fresh rows inside one transaction.
- Load all vectors once per process into a single `numpy` matrix; search with `matrix @ query_vector`. For a single book this is milliseconds.
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

Retrieval (retrieve wide, then rerank narrow):

1. Embed the question with the **same** embedder used for the book. With MiniLM, no prefix is needed.
2. **Vector search:** cosine similarity against all chunk vectors, take `candidate_k` (default 30).
3. **Keyword search:** BM25 over the chunk text, take `candidate_k`.
4. **Hybrid merge:** normalize BM25 scores to `[0, 1]`, then `score = alpha * cosine + (1 - alpha) * bm25_normalized`. This handles both paraphrased questions and exact names.
5. **Rerank:** run the merged candidates through a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and keep the best `top_k` (default 5). This is a separate model — do **not** rerank by re-embedding with the same embedder.
6. **Threshold + dedup:** drop results below `MIN_SCORE`; merge adjacent chunks from the same chapter/page so the LLM does not receive five near-duplicates. If nothing clears the threshold, return “That does not appear in this book.” and skip the LLM call.
7. Only the final `top_k` chunks are sent to the answer model. Keep total context bounded (~5 × 1000 chars).

**Context budget (LLM prompt):**

```text
system prompt        ~300–500 tokens
question             ~50–200
retrieved chunks     ~1,500–3,000   (top 5)
conversation history ~500–1,000     (recent turns only; summarize older)
output reserve       ~500–1,000
------------------------------------
total                ~4–6K tokens   (even if the model allows 1M)
```

Do not fill the window just because it is large. More context raises cost and latency and can bury the best passage in the middle. Put the strongest chunks first/last, deduplicate before sending, and skip the LLM call entirely when nothing clears `MIN_SCORE`.

Answering, two modes:

- `extractive` **(default, no API key):** return the best passage plus the source list. Honest, fast, and useful for testing retrieval.
- `gemini` / `openai` **(or any OpenAI-compatible endpoint):** send numbered passages and ask for an answer with `[1]`-style citations. Prompt template:

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

Return `{answer, sources[]}` where each source includes chapter, page range, score, and an excerpt. Citations are the difference between a demo and a usable RAG app.

**Acceptance check:** ask a question whose answer you know is on a specific page; the top source should be that chapter and page. Ask something unrelated; the app should refuse rather than invent.

**Evaluating retrieval quality:** build a small set of `question -> expected chunk/page` pairs. Start with **Recall@5** (did the right chunk appear in the top 5?) and **MRR** (how high did it rank?). Use **NDCG@5** when you assign graded relevance labels (perfect / partial / irrelevant) or when several chunks are relevant — it rewards putting the best ones first, not just including them. NDCG@k is an evaluation metric, not an embedding model; the embedder stays `all-MiniLM-L6-v2`.

---

## 10. FastAPI layer

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
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
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
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

- In `app/main.py`, use a `lifespan` handler to create tables and load the embedding model **once**. Loading a model per request is the most common performance mistake.
- File uploads: `UploadFile`; validate extension and size; save to `data/raw/<sha>.pdf`; then call the ingest service.
- CPU-heavy steps block the event loop. For a single book it is acceptable to run them in `fastapi.concurrency.run_in_threadpool`, or use `BackgroundTasks`plus `GET /api/books/{id}/status` if you want progress.
- Keep API schemas (`app/schemas/`) separate from the internal RAG dataclasses. That lets you change internal fields without breaking clients.
- Return meaningful status codes: 400 unsupported file, 409 book already indexed (or replace it), 422 bad query, 500 pipeline failure.
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

For generated answers, set `LLM_BACKEND=gemini` and `GEMINI_API_KEY` in `.env`; keep `LLM_BACKEND=extractive` for offline tests and for checking retrieval before spending API calls.

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

 1. **Environment** — create venv, install requirements, copy `.env.example` to `.env`.
 2. **Config** — write `app/core/config.py`; confirm `Settings()` prints the expected values.
 3. **Database** — write `app/db/database.py` and `models.py`; create tables.
 4. **Step 1** — write `step1_ingest.py`; run it from a scratch script on your book; inspect the chapters JSONL.
 5. **Step 2** — write `step2_chunk.py`; inspect chunk count, lengths, overlap.
 6. **Step 3** — write `step3_embed.py`; assert vector shape and norm.
 7. **Step 4** — write `step4_store.py`; run the SQL acceptance checks.
 8. **Step 5** — write `step5_answer.py`; test retrieval with `extractive` mode before adding any LLM key.
 9. **Pipeline** — write `pipeline.py` to run 1–4 and `scripts/ingest_book.py`to call it.
10. **Schemas + services** — define request/response models and move the orchestration out of scripts into `services/`.
11. **API** — implement routes and `main.py`; start uvicorn; ingest and query through `/docs`.
12. **Tests** — add the tests below; run `pytest -q`.
13. **Commit** — initialize git and commit (section 13).

A useful rule: **do not write FastAPI code until the five steps work from a script.** HTTP is the easy part; retrieval quality is the hard part.

---

## 13. Tests to write

- `tests/test_ingest.py` — a tiny fixture `.txt` produces the expected number of chapters and keeps page numbers.
- `tests/test_chunking.py` — chunks respect `max_chars`, contain every sentence, and overlap.
- `tests/test_retrieval.py` — index a 3–4 chapter fixture; a question about a distinctive fact returns the right chapter; an unrelated question returns no confident hit.
- `tests/test_rerank.py` (optional) — a fake or tiny reranker reorders candidates as expected; assert the final `top_k` and that dedup/threshold logic runs before the LLM call.
- `tests/test_api.py` — use `fastapi.testclient.TestClient` with a temporary `BOOKRAG_DATA_DIR`; ingest a fixture, query it, assert `sources` is non-empty and the answer contains the expected fact.
- Keep tests offline: use the hashing/`extractive` path or monkeypatch the embedder; never call a paid API in CI.

```bash
pytest -q
```

---

## 14. Commit

Generated data (`.env`, `data/raw/*`, `data/processed/*`, `data/index/*`) is already ignored. `.gitkeep` files keep the empty folders in the repository.

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

If your book is public domain or you have the rights, remove the `data/raw/*` ignore rule before committing it; otherwise keep it local and document where users should drop their own copy.

---

## 15. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| PDF ingests 0 characters | Scanned image PDF | Run OCR first, e.g. `ocrmypdf in.pdf out.pdf` |
| `dimension mismatch` on query | Index built with another embedding model | Re-embed and re-store, or restore the original model |
| First query takes 30+ seconds | Model download / lazy load | Preload embedder **and** reranker in `lifespan`; warm up with one dummy query |
| Answers cite the wrong chapter | Chunks crossed chapter boundaries or page metadata lost | Keep chapter/page on every chunk; never drop page numbers in step 1 |
| Five near-identical sources | Adjacent chunks overlap | Deduplicate by chapter + nearby page, or use MMR |
| Answers sound invented | LLM allowed outside knowledge | Require citations; enforce `MIN_SCORE`; say “not in the book” otherwise |
| Ingest is slow | Embedding on CPU in the request path | Batch, or move steps 1–4 to a background job |
| Query is slow after adding rerank | Cross-encoder runs over too many candidates | Lower `CANDIDATE_K` (e.g. 20) or run the reranker on a separate service |
| Keyword results dominate hybrid scores | Raw BM25 not normalized | Normalize BM25 to `[0, 1]` before combining with `hybrid_alpha` |
| Gemini returns 429 / quota errors | Free-tier rate limits | Cache results, batch queries, or fall back to `extractive` mode |
| Query retrieval quality dropped after model swap | Query prefix or model changed | MiniLM needs no prefix; if you use BGE, add its query instruction consistently |
| Retrieval got worse after using 1000-char chunks | MiniLM silently truncates past 256 tokens | Lower `CHUNK_SIZE` to ~700–800 or switch to a 512-token embedder |
| Answers ignore the best passage | Too much context / best chunk buried in the middle | Reduce `TOP_K`, put strong chunks first/last, remove near-duplicates |
| Token usage higher than expected | Overlapping chunks duplicated in the prompt | Deduplicate by chapter/page before the LLM call; log token counts |
| Re-upload duplicates chunks | No idempotency key | Delete by `sha256` before inserting |
| SQLite `database is locked` | Concurrent writers | Serialize ingest; one job at a time, or move to Postgres |

---

## 16. When to outgrow this design

The five steps stay the same; only the pieces behind them change:

- **Better retrieval:** the default already reranks with `ms-marco-MiniLM-L-6-v2`; at scale, move that reranker to HF Text Embeddings Inference / Infinity and increase `candidate_k` to 50.
- **Better embeddings:** swap MiniLM for a stronger model (BGE-M3, Qwen3-Embedding) only after measuring Recall@5; a new dimension means re-embedding and re-storing the whole book.
- **More books / users:** replace SQLite/NumPy with Postgres + pgvector or Qdrant; add metadata filters (`book_id`, chapter, page); move embeddings and reranking to dedicated inference services.
- **Production ingest:** queue jobs (Celery/RQ/Arq), report progress, retry failures.
- **Streaming answers:** stream tokens over SSE and send sources first.
- **Evaluation:** build a small question → expected-page test set and measure Recall@5 / MRR / NDCG@5 before changing chunk size or models. Cache query embeddings and final answers for repeated questions.
- **Token efficiency:** log prompt/completion tokens per query; cap retrieved context tokens, summarize old conversation turns instead of replaying them, and reuse cached retrieval results for repeated questions.

Keep the five-step boundaries. They make every failure easy to locate: bad text, bad chunks, bad vectors, bad storage, or bad prompting.