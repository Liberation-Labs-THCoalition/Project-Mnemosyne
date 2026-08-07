"""Regression tests for the symbolic (significance) ranking query.

Precious integration audit finding #3: the hybrid-retrieval symbolic search
rewarded LOW numeric significance, inverting the documented convention
(1 = low/ephemeral, 10 = high/core).  These tests execute the exact query
shipped in ``kintsugi.api.routes.memory`` against a real database to prove
that high-significance memories rank first and receive the higher scores.
"""

from __future__ import annotations

import uuid

from sqlalchemy import create_engine, text

from kintsugi.api.routes.memory import SYMBOLIC_SQL


def _make_engine():
    """In-memory SQLite engine with a minimal memory_units table.

    Only the columns referenced by SYMBOLIC_SQL are needed; the query itself
    is dialect-portable (CAST instead of ::float) precisely so it can run on
    the SQLite seed tier as well as Postgres.
    """
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE memory_units (
                id TEXT PRIMARY KEY,
                org_id TEXT NOT NULL,
                content TEXT NOT NULL,
                significance INTEGER NOT NULL,
                memory_layer TEXT NOT NULL DEFAULT 'core',
                created_at TEXT NOT NULL
            )
        """))
    return engine


def _insert(conn, org_id: str, content: str, significance: int,
            created_at: str = "2026-08-01T00:00:00") -> None:
    conn.execute(
        text("""
            INSERT INTO memory_units
                (id, org_id, content, significance, memory_layer, created_at)
            VALUES
                (:id, :org_id, :content, :significance, 'core', :created_at)
        """),
        {
            "id": str(uuid.uuid4()),
            "org_id": org_id,
            "content": content,
            "significance": significance,
            "created_at": created_at,
        },
    )


class TestSymbolicRanking:
    def test_high_significance_ranks_above_low(self):
        """A significance-9 memory must outrank a significance-2 memory."""
        engine = _make_engine()
        org = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert(conn, org, "ephemeral aside", significance=2)
            _insert(conn, org, "core organizational value", significance=9)

        with engine.connect() as conn:
            rows = conn.execute(SYMBOLIC_SQL, {"org_id": org, "limit": 10}).fetchall()

        assert len(rows) == 2
        assert rows[0].content == "core organizational value"
        assert rows[0].significance == 9
        assert rows[1].significance == 2
        assert rows[0].score > rows[1].score

    def test_scores_track_significance_across_full_range(self):
        """Result order and score are monotone in significance for 1..10."""
        engine = _make_engine()
        org = str(uuid.uuid4())
        # Insert out of order to prove ordering comes from the query.
        with engine.begin() as conn:
            for sig in [4, 9, 1, 7, 10, 2, 6, 3, 8, 5]:
                _insert(conn, org, f"memory sig={sig}", significance=sig)

        with engine.connect() as conn:
            rows = conn.execute(SYMBOLIC_SQL, {"org_id": org, "limit": 10}).fetchall()

        sigs = [r.significance for r in rows]
        assert sigs == sorted(sigs, reverse=True), (
            "symbolic search must order by significance DESC (10 = core first)"
        )
        for row in rows:
            assert row.score == row.significance / 10.0
        scores = [r.score for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_ties_broken_by_recency(self):
        """Equal significance falls back to newest-first ordering."""
        engine = _make_engine()
        org = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert(conn, org, "older", 5, created_at="2026-07-01T00:00:00")
            _insert(conn, org, "newer", 5, created_at="2026-08-01T00:00:00")

        with engine.connect() as conn:
            rows = conn.execute(SYMBOLIC_SQL, {"org_id": org, "limit": 10}).fetchall()

        assert [r.content for r in rows] == ["newer", "older"]

    def test_org_scoping(self):
        """Only the requesting org's memories are returned."""
        engine = _make_engine()
        org_a = str(uuid.uuid4())
        org_b = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert(conn, org_a, "org-a memory", 8)
            _insert(conn, org_b, "org-b memory", 8)

        with engine.connect() as conn:
            rows = conn.execute(SYMBOLIC_SQL, {"org_id": org_a, "limit": 10}).fetchall()

        assert [r.content for r in rows] == ["org-a memory"]

    def test_limit_keeps_highest_significance(self):
        """When limited, the surviving rows are the most significant ones."""
        engine = _make_engine()
        org = str(uuid.uuid4())
        with engine.begin() as conn:
            for sig in range(1, 11):
                _insert(conn, org, f"memory sig={sig}", significance=sig)

        with engine.connect() as conn:
            rows = conn.execute(SYMBOLIC_SQL, {"org_id": org, "limit": 3}).fetchall()

        assert [r.significance for r in rows] == [10, 9, 8]
