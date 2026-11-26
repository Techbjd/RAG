from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import get_settings
from app.db.database import create_db_engine, create_session_factory, init_db

from app.api.routes import health, rag

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = create_db_engine(settings)
    init_db(engine, settings)
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = create_session_factory(engine)
    yield
    engine.dispose()

app = FastAPI(title="BOOK RAG", lifespan=lifespan)

app.include_router(health.router, tags=["health"])
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(rag.router, prefix="/api", tags=["rag"])


@app.get("/v1/models", include_in_schema=False)
def openai_models():
    """Satisfy local OpenAI-compatible client discovery."""
    return {"object": "list", "data": []}
