# Book RAG

A local retrieval-augmented generation pipeline for books. It ingests a
book, splits it into chapters and overlapping chunks, embeds the chunks,
stores them in Postgres + pgvector, and answers questions with cited passages.

The database runs in Docker. All book-rag tables live in the `book_rag`
schema of the `bookrag` database, leaving `public` untouched.

## Pipeline

| Step | Module | Purpose |
| --- | --- | --- |
| 1. Ingest | `app/rag/step1_ingest.py` | Read `.pdf`, `.txt`, `.md`, or `.json` and produce cleaned, page-aware chapters |
| 2. Chunk | `app/rag/step2_chunk.py` | Split chapters into overlapping passages with citation metadata |
| 3. Embed | `app/rag/step3_embed.py` | Encode chunks with a frozen sentence-transformer model |
| 4. Store | `app/rag/step4_store.py` | Persist documents, chunks, and vectors in Postgres (`book_rag` schema) |
| 5. Answer | `app/rag/step5_answer.py` | Hybrid retrieval, reranking, thresholding, and grounded answers |

`app/rag/pipeline.py` runs Steps 1–4. Step 5 is the read path and runs once
per question.

## Project status

- Docker + pgvector database is running in the `book_rag` schema.
- Steps 1–5 are implemented and the full pipeline works end to end.
- The FastAPI layer exposes book upload/indexing and query endpoints.
- CLI scripts are not implemented yet.

## Requirements

- Python 3.11 or newer
- Docker with Compose
- A CPU is sufficient; the default models run locally

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first embedding or reranking call downloads the configured model.

## Database

Start the pgvector container from the project root, then let `init_db()`
create the extension, schema, and tables:

```bash
docker compose up -d
python -c "from app.db.database import create_db_engine, init_db; init_db(create_db_engine())"
```

Confirm the extension and schema:

```bash
docker compose exec db psql -U bookrag -d bookrag \
  -c "select extversion from pg_extension where extname='vector';"
```

`compose.yaml` binds Postgres to `127.0.0.1:5433` and uses the named volume
`bookrag_pgdata`. `init_db()` creates the pgvector extension, the `book_rag`
schema, and all tables.

## Configuration

Copy the example environment file and adjust values as needed:

```bash
cp .env.example .env
```

Settings are loaded by `app/core/config.py`.

| Variable | Default | Description |
| --- | --- | --- |
| `DATA_DIR` | `./data` | Base directory for raw, processed, and index data |
| `UPLOAD_SUFFIXES` | `.pdf,.txt,.md,.json` | Extensions accepted by `POST /api/books` |
| `MAX_UPLOAD_MB` | `50` | Maximum upload size in megabytes (`0` disables the limit) |
| `DATABASE_URL` | `postgresql+psycopg://bookrag:bookrag_dev@127.0.0.1:5433/bookrag` | SQLAlchemy database URL |
| `DATABASE_SCHEMA` | `book_rag` | Schema used by all book-rag tables |
| `CHUNK_SIZE` | `1000` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between adjacent chunks |
| `EMBEDDING_BACKEND` | `sentence-transformers` | Embedding backend |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Frozen embedding model |
| `EMBEDDING_DIM` | `384` | Must match the model output dimension |
| `EMBEDDING_BATCH` | `64` | Encoder batch size |
| `CANDIDATE_K` | `30` | Candidates retrieved before reranking |
| `TOP_K` | `5` | Final passages used for the answer |
| `HYBRID_ALPHA` | `0.6` | Vector weight in hybrid retrieval (`0` = BM25 only, `1` = vectors only) |
| `MIN_SCORE` | `0.20` | Minimum reranker score |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `LLM_BACKEND` | `extractive` | `extractive` or `gemini` |
| `LLM_MODEL` | `gemini-2.0-flash` | Generation model |
| `GEMINI_API_KEY` | empty | Required when `LLM_BACKEND=gemini` |

`LLM_BACKEND=extractive` returns the best matching passage without calling an
external API. It is the default for offline use and retrieval testing.

## Usage

Run the indexing pipeline from Python:

```python
from app.core.config import get_settings
from app.rag.pipeline import index_book

settings = get_settings()
result = index_book("data/raw/book.pdf", settings=settings)

print(result.book.title)
print(len(result.chunks))
print(result.embedded.vectors.shape)
```

Run a question against the stored index:

```python
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import create_db_engine, create_session_factory
from app.rag.step5_answer import answer_question

settings = get_settings()
engine = create_db_engine(settings)
factory = create_session_factory(engine)

with factory() as session:
    answer = answer_question("Who built the tide clock?", session=session)
    print(answer.answer)
    for source in answer.sources:
        print(source.chapter, source.page_start, source.page_end)
```

### HTTP API

Start the server:

```bash
uvicorn app.main:app --reload
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Database health check |
| `POST` | `/api/books` | Upload a book (multipart `file`) and run Steps 1–4 |
| `GET` | `/api/books` | List indexed books |
| `POST` | `/api/query` | Ask a question (Step 5) |

```bash
curl -F "file=@data/raw/book.pdf" http://127.0.0.1:8000/api/books

curl -X POST http://127.0.0.1:8000/api/query \
  -H 'content-type: application/json' \
  -d '{"question":"Who built the tide clock?","top_k":5}'
```

Uploads are saved under `DATA_DIR/raw/`. The accepted extensions, size limit,
and `top_k` default all come from `.env` through `app/core/config.py`.

## Project layout

```text
compose.yaml      Postgres + pgvector container
app/
  api/            FastAPI routes (health, books, query)
  core/config.py  Typed settings from environment
  db/database.py  Engine, sessions, schema initialization
  db/models.py    SQLAlchemy schema
  rag/
    step1_ingest.py
    step2_chunk.py
    step3_embed.py
    step4_store.py
    step5_answer.py
    retrieval.py  Hybrid vector + BM25 search
    rerank.py     Cross-encoder reranking
    pipeline.py   Steps 1-4 orchestration
  schemas/        API request/response models
  services/       Service layer (reserved)
data/
  raw/            Source books (git-ignored)
  processed/      Generated artifacts (git-ignored)
tests/            Offline unit tests
```

## Data and generated files

- Put source books in `data/raw/`.
- Generated files under `data/processed/` are ignored by Git.
- Postgres data lives in the `bookrag_pgdata` Docker volume, outside the repo.
- `.env` is ignored; use `.env.example` as the template.

## Tests

```bash
pytest -q
```

The test suite is offline. Retrieval and reranking tests use fake scores and
do not download model weights. Run it as `python -m pytest -q` (or plain
`pytest -q`, which uses the `pythonpath = .` setting in `pytest.ini`).

## Known limitations

- There is no CLI yet; use the HTTP API or the Python examples above.
- Step 4 needs Postgres (pgvector), so its tests are not part of the offline
  suite.
- Embeddings are recomputed for the whole book on every re-upload; there is no
  incremental indexing.
