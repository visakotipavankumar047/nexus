from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from starlette.concurrency import run_in_threadpool
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.utils.errors import DatabaseException


@lru_cache
def get_engine() -> Engine:
    url = get_settings().database_url.get_secret_value()
    if not url:
        raise DatabaseException("DATABASE_URL is not set")
    return create_engine(
        url.replace("postgresql://", "postgresql+psycopg://", 1),
        pool_pre_ping=True,
        connect_args={"connect_timeout": 10, "options": "-c statement_timeout=10000"},
    )


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(get_engine(), expire_on_commit=False)


@contextmanager
def session() -> Iterator[Session]:
    """One transaction: commits on success, rolls back and raises DatabaseException on DB errors."""
    try:
        with _session_factory().begin() as s:
            yield s
    except SQLAlchemyError as e:
        raise DatabaseException() from e


async def run_in_session[T](fn: Callable[[Session], T]) -> T:
    """Runs fn in one transaction on a worker thread so blocking DB I/O never stalls the event loop."""
    def work() -> T:
        with session() as s:
            return fn(s)
    return await run_in_threadpool(work)
