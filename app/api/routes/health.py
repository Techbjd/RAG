from fastapi import APIRouter,Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_session

router=APIRouter()


@router.get("/health")
def health(session: Session = Depends(get_session)):
    session.execute(text("select 1"))
    return {"status": "ok", "database": "ok"}
