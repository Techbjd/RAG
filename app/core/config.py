from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- App ---
    app_name: str = "book-rag"
    data_dir: str = "./data"
    database_url: str = (
        "postgresql+psycopg://bookrag:bookrag_dev@127.0.0.1:5433/bookrag"
    )
    # Dedicated schema for this project; "public" is left untouched.
    database_schema: str = "book_rag"

    # --- API: uploads ---
    # Comma-separated allowlist of file extensions accepted by POST /api/books.
    upload_suffixes: str = ".pdf,.txt,.md,.json"
    # Hard limit per uploaded file; larger uploads are rejected with 413.
    max_upload_mb: int = 50

    # --- Step 2: chunking ---
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Step 3: embeddings ---
    # sentence-transformers (local, default) or openai
    embedding_backend: str = "sentence-transformers"
    # all-MiniLM-L6-v2: 384-dim, no query prefix, fast on CPU
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch: int = 64

    # --- Step 5: retrieval + rerank + answering ---
    # Candidates retrieved before reranking (retrieve wide, rerank narrow)
    candidate_k: int = 30
    # Final chunks sent to the LLM after reranking
    top_k: int = 5
    # 0 = BM25 only, 1 = vectors only
    hybrid_alpha: float = 0.6
    min_score: float = 0.20
    # Cross-encoder reranker (separate model; does not affect stored vectors)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Step 5: answer generation ---
    # extractive (no API key), gemini, or openai
    llm_backend: str = "extractive"
    llm_model: str = "gemini-2.0-flash"
    gemini_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    @property
    def raw_dir(self) -> Path:
        """Directory where uploaded books are stored."""
        return Path(self.data_dir) / "raw"

    @property
    def allowed_upload_suffixes(self) -> set[str]:
        """Parse ``upload_suffixes`` into a lowercase set of extensions."""
        return {
            suffix.strip().lower()
            for suffix in self.upload_suffixes.split(",")
            if suffix.strip()
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
