"""Functional tests for /api/agent/message temporal-event deduplication.

Precious integration audit finding #2 (note): repeated POSTs of the same
message created duplicate temporal_memories rows.  The endpoint now treats
an identical (org, category, message) event inside
``TEMPORAL_DEDUP_WINDOW_SECONDS`` as a duplicate delivery and returns the
original event instead of inserting a new row.

Runs the real route against an in-memory SQLite database.  The tables are
created with raw DDL because the ORM's Postgres-native column types (JSONB)
have no SQLite DDL rendering; the ORM handles binds/results fine.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from kintsugi.api.routes.agent import TEMPORAL_DEDUP_WINDOW_SECONDS, router
from kintsugi.db import get_session
from kintsugi.models.base import Organization, TemporalMemory

_DDL_ORGANIZATIONS = """
CREATE TABLE organizations (
    id CHAR(32) PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    org_type VARCHAR(64) NOT NULL,
    created_at TIMESTAMP,
    values_json TEXT,
    bdi_json TEXT
)
"""

_DDL_TEMPORAL_MEMORIES = """
CREATE TABLE temporal_memories (
    id CHAR(32) PRIMARY KEY,
    org_id CHAR(32) NOT NULL,
    category VARCHAR(128) NOT NULL,
    message TEXT NOT NULL,
    metadata_json TEXT,
    created_at TIMESTAMP
)
"""


@pytest.fixture
async def harness():
    """(client, org_id, sessionmaker) wired to a fresh in-memory SQLite DB."""
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.execute(text(_DDL_ORGANIZATIONS))
        await conn.execute(text(_DDL_TEMPORAL_MEMORIES))

    maker = async_sessionmaker(engine, expire_on_commit=False)

    org_id = uuid.uuid4()
    async with maker() as session:
        session.add(Organization(id=org_id, name="Test Org"))
        await session.commit()

    app = FastAPI()
    app.include_router(router)

    async def _override_session():
        async with maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_session] = _override_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, str(org_id), maker

    await engine.dispose()


async def _count_events(maker) -> int:
    async with maker() as session:
        result = await session.execute(text("SELECT COUNT(*) FROM temporal_memories"))
        return result.scalar_one()


class TestAgentMessageDedup:
    async def test_repeated_message_reuses_event(self, harness):
        client, org_id, maker = harness
        body = {"message": "status check ping", "org_id": org_id}

        first = await client.post("/api/agent/message", json=body)
        second = await client.post("/api/agent/message", json=body)

        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["deduplicated"] is False
        assert second.json()["deduplicated"] is True
        assert second.json()["temporal_event_id"] == first.json()["temporal_event_id"]
        assert await _count_events(maker) == 1

    async def test_distinct_messages_create_distinct_events(self, harness):
        client, org_id, maker = harness

        first = await client.post(
            "/api/agent/message", json={"message": "first note", "org_id": org_id}
        )
        second = await client.post(
            "/api/agent/message", json={"message": "second note", "org_id": org_id}
        )

        assert first.json()["deduplicated"] is False
        assert second.json()["deduplicated"] is False
        assert (
            second.json()["temporal_event_id"] != first.json()["temporal_event_id"]
        )
        assert await _count_events(maker) == 2

    async def test_duplicate_outside_window_creates_new_event(self, harness):
        client, org_id, maker = harness
        message = "daily standup summary"

        # Plant an identical event well outside the dedup window.
        stale_time = datetime.now(timezone.utc) - timedelta(
            seconds=TEMPORAL_DEDUP_WINDOW_SECONDS * 4
        )
        async with maker() as session:
            session.add(
                TemporalMemory(
                    org_id=uuid.UUID(org_id),
                    category="interaction",
                    message=message,
                    metadata_json={},
                    created_at=stale_time,
                )
            )
            await session.commit()

        response = await client.post(
            "/api/agent/message", json={"message": message, "org_id": org_id}
        )

        assert response.status_code == 200
        assert response.json()["deduplicated"] is False
        assert await _count_events(maker) == 2

    async def test_dedup_scoped_to_org(self, harness):
        client, org_id, maker = harness

        other_org = uuid.uuid4()
        async with maker() as session:
            session.add(Organization(id=other_org, name="Other Org"))
            await session.commit()

        body_a = {"message": "shared phrasing", "org_id": org_id}
        body_b = {"message": "shared phrasing", "org_id": str(other_org)}

        first = await client.post("/api/agent/message", json=body_a)
        second = await client.post("/api/agent/message", json=body_b)

        assert first.json()["deduplicated"] is False
        assert second.json()["deduplicated"] is False
        assert await _count_events(maker) == 2
