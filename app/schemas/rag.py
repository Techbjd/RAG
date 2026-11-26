"""Request/response models for the RAG HTTP API."""

from pydantic import BaseModel, Field


class IngestOut(BaseModel):
    """Summary of one book indexed by POST /api/books."""

    document_id: str
    title: str
    chapters: int
    chunks: int
    dim: int
    replaced: bool


class QueryRequest(BaseModel):
    """Body for POST /api/query."""

    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0)


class SourceOut(BaseModel):
    """A cited passage returned with an answer."""

    chunk_id: str
    chapter: str
    page_start: int | None
    page_end: int | None
    score: float
    text: str


class AnswerOut(BaseModel):
    """Body returned by POST /api/query."""

    answer: str
    sources: list[SourceOut]


class DocumentOut(BaseModel):
    """One indexed book in GET /api/books."""

    id: str
    title: str
    source_path: str
