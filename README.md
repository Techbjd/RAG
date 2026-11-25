# Book RAG

A local retrieval-augmented generation pipeline for books. It ingests a
book, splits it into chapters and overlapping chunks, embeds the chunks,
stores them in SQLite, and answers questions with cited passages.

## Pipeline

| Step | Module | Purpose |
| --- | --- | --- |
| 1. Ingest | `app/rag/step1_ingest.py` | Read `.pdf`, `.txt`, `.md`, or `.json` and produce cleaned, page-aware chapters |
| 2. Chunk | `app/rag/step2_chunk.py` | Split chapters into overlapping passages with citation metadata |
| 3. Embed | `app/rag/step3_embed.py` | Encode chunks with a frozen sentence-transformer model |
| 4. Store | `app/rag/step4_store.py` | Persist documents, chunks, and vectors in SQLite |
| 5. Answer | `app/rag/step5_answer.py` | Hybrid retrieval, reranking, thresholding, and grounded answers |

`app/rag/pipeline.py` runs Steps 1–4. Step 5 is the read path and runs once
per question.

## Project status

- Steps 1–3 are implemented and covered by tests.
- Step 4 has the schema and helpers, but the row-writing part of
  `store_index()` is still incomplete.
- Step 5 is implemented, but it cannot return meaningful sources until Step 4
  writes chunks and vectors.
- The FastAPI layer and CLI scripts are not implemented yet.

## Requirements

- Python 3.11 or newer
- A CPU is sufficient; the default models run locally

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first embedding or reranking call downloads the configured model.

## Configuration

Copy the example environment file and adjust values as needed:

```bash
cp .env.example .env
```

Settings are loaded by `app/core/config.py`.

| Variable | Default | Description |
| --- | --- | --- |
| `DATA_DIR` | `./data` | Base directory for raw, processed, and index data |
| `DATABASE_URL` | `sqlite:///./data/index.db` | SQLAlchemy database URL |
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
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.rag.step5_answer import answer_question

settings = get_settings()
engine = create_engine(settings.database_url)

with Session(engine) as session:
    answer = answer_question("Who built the tide clock?", session=session)
    print(answer.answer)
    for source in answer.sources:
        print(source.chapter, source.page_start, source.page_end)
```

These end-to-end examples require the pending Step 4 row writes.

## Project layout

```text
app/
  api/            FastAPI routes (not implemented)
  core/config.py  Typed settings from environment
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
  schemas/        API schemas (not implemented)
  services/       Service layer (not implemented)
data/
  raw/            Source books (git-ignored)
  processed/      Generated artifacts (git-ignored)
  index/          SQLite index (git-ignored)
tests/            Offline unit tests
```

## Data and generated files

- Put source books in `data/raw/`.
- Generated files under `data/processed/` and `data/index/` are ignored by Git.
- `.env` is ignored; use `.env.example` as the template.

## Tests

```bash
pytest -q
```

The test suite is offline. Retrieval and reranking tests use fake scores and
do not download model weights.

## Known limitations

- `step4_store.validate_embedding_result()` and the row inserts in
  `store_index()` are incomplete.
- SQLite auto-increment for the `Chapter` and `Chunk` primary keys still needs
  an `INTEGER` variant before inserts work.
- There is no HTTP API or CLI yet.
