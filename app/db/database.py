"""Engine, sessions, and schema setup for the book-rag Postgres database.

The database itself is created by Docker Compose; this module makes sure the
pgvector extension and the dedicated ``book_rag`` schema exist, then creates
the tables defined in ``app.db.models`` inside that schema.
"""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.db.models import Base


def _schema_name(settings: Settings) -> str:
    """Return the configured schema, rejecting anything not a plain name."""
    schema = settings.database_schema
    if not schema.isidentifier() or schema.startswith("pg_"):
        raise ValueError(f"invalid database_schema={schema!r}")
    return schema


def create_db_engine(settings: Settings | None = None) -> Engine:
    """Build the engine and pin every connection to the project schema.

    ``search_path`` puts ``book_rag`` first so unqualified table names resolve
    to it, while ``public`` stays available for the pgvector extension.
    """
    if settings is None:
        settings = get_settings()

    connect_args: dict[str, object] = {}
    pool_kwargs: dict[str, object] = {}
    if make_url(settings.database_url).get_backend_name() == "postgresql":
        if (
            settings.database_pool_size < 1
            or settings.database_max_overflow < 0
            or settings.database_pool_timeout < 1
            or settings.database_pool_recycle < -1
        ):
            raise ValueError("invalid database pool settings")

        connect_args["options"] = (
            f"-csearch_path={_schema_name(settings)},public"
        )
        pool_kwargs = {
            "pool_size": settings.database_pool_size,
            "max_overflow": settings.database_max_overflow,
            "pool_timeout": settings.database_pool_timeout,
            "pool_recycle": settings.database_pool_recycle,
        }

    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args=connect_args,
        **pool_kwargs,
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory that keeps objects usable after commit."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine, settings: Settings | None = None) -> None:
    """Create the extension, the schema, and all book-rag tables."""
    if settings is None:
        settings = get_settings()

    if make_url(engine.url).get_backend_name() == "postgresql":
        schema = _schema_name(settings)
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.execute(
                text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            )

    Base.metadata.create_all(engine)
