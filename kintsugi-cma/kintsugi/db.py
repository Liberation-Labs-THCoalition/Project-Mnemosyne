"""Async SQLAlchemy database setup.

The engine is created lazily on first use (not at import time) so that
modules importing :class:`Base`, models, or route handlers do not require
a database driver to be installed.  ``engine`` and ``async_session``
remain importable as module attributes for backward compatibility.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from kintsugi.config.settings import settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.DATABASE_URL,
            echo=False,
            pool_pre_ping=True,
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide sessionmaker, creating it on first use."""
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _sessionmaker


def __getattr__(name: str):
    # Backward-compatible module attributes (`from kintsugi.db import engine`).
    if name == "engine":
        return get_engine()
    if name == "async_session":
        return get_sessionmaker()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class Base(DeclarativeBase):
    pass


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def set_org_context(session: AsyncSession, org_id: str) -> None:
    """Bind the current transaction to an organization for RLS enforcement.

    On PostgreSQL this sets the transaction-local ``app.current_org_id``
    setting that the row-level-security policies (migration 002) check.
    Every request-scoped transaction that touches org-scoped tables must
    call this first, otherwise RLS makes those tables appear empty and
    rejects writes.

    On non-PostgreSQL backends (the SQLite seed tier) this is a no-op:
    SQLite has no row-level security, so isolation there relies on the
    application-level ``org_id`` filters alone.
    """
    bind = session.get_bind()
    if bind.dialect.name != "postgresql":
        return
    await session.execute(
        text("SELECT set_config('app.current_org_id', :org_id, true)"),
        {"org_id": str(org_id)},
    )
