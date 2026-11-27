"""Book RAG router mounted at ``/api`` by ``app.main``.

The endpoints are split into small route modules; this module only aggregates
them so the mounting point stays unchanged.
"""

from fastapi import APIRouter

from app.api.routes import books, query

router = APIRouter()
router.include_router(books.router)
router.include_router(query.router)
