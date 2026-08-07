"""Row-level security on the live memory tables.

Audit finding #12: RLS existed only in the standalone ``org_memories``
module (kintsugi/memory/org_isolation.py) while the live API operates on
``memory_units`` and its sibling tables from migration 001, which had no
RLS at all.  This migration applies the same policy pattern to the tables
the API actually uses.

Pattern (mirrors ORG_MEMORIES_SCHEMA):
  - ENABLE + FORCE row level security (FORCE so the table-owning app role
    is also subject to the policies).
  - Per-operation policies checking
    ``org_id::text = current_setting('app.current_org_id', true)``.
  - Child tables without an org_id column (memory_embeddings,
    memory_lexical, memory_metadata) are scoped through their parent
    memory_units row via EXISTS, which is itself RLS-filtered.

Operational contract: every transaction touching these tables must first
call ``kintsugi.db.set_org_context(session, org_id)`` (the API routes do).
A transaction without the setting sees zero rows and cannot write.

``organizations`` intentionally stays outside RLS: it is the tenant
registry the API consults to validate org ids before binding the context.

PostgreSQL only — the SQLite seed tier has no RLS mechanism; this
migration is a no-op there (documented in COMPONENT_MANIFEST.md).

Revision ID: 002
Revises: 001
Create Date: 2026-08-06
"""
from typing import Sequence, Union

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Tables carrying an org_id column.
_ORG_TABLES = [
    "memory_units",
    "temporal_memories",
    "memory_archives",
    "intent_capsules",
    "shield_constraints",
]

# Child tables scoped through memory_units(memory_id -> id).
_CHILD_TABLES = [
    "memory_embeddings",
    "memory_lexical",
    "memory_metadata",
]

_ORG_CHECK = "org_id::text = current_setting('app.current_org_id', true)"

_CHILD_CHECK = (
    "EXISTS (SELECT 1 FROM memory_units mu "
    "WHERE mu.id = memory_id "
    "AND mu.org_id::text = current_setting('app.current_org_id', true))"
)


def _policies(table: str, check: str) -> list[str]:
    return [
        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
        f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",
        f"CREATE POLICY rls_{table}_select ON {table} FOR SELECT USING ({check})",
        f"CREATE POLICY rls_{table}_insert ON {table} FOR INSERT WITH CHECK ({check})",
        (
            f"CREATE POLICY rls_{table}_update ON {table} FOR UPDATE "
            f"USING ({check}) WITH CHECK ({check})"
        ),
        f"CREATE POLICY rls_{table}_delete ON {table} FOR DELETE USING ({check})",
    ]


def _drop_policies(table: str) -> list[str]:
    return [
        f"DROP POLICY IF EXISTS rls_{table}_select ON {table}",
        f"DROP POLICY IF EXISTS rls_{table}_insert ON {table}",
        f"DROP POLICY IF EXISTS rls_{table}_update ON {table}",
        f"DROP POLICY IF EXISTS rls_{table}_delete ON {table}",
        f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY",
        f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY",
    ]


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return  # SQLite seed tier: no RLS support; app-level scoping only.

    for table in _ORG_TABLES:
        for stmt in _policies(table, _ORG_CHECK):
            op.execute(stmt)

    for table in _CHILD_TABLES:
        for stmt in _policies(table, _CHILD_CHECK):
            op.execute(stmt)


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    for table in _CHILD_TABLES + _ORG_TABLES:
        for stmt in _drop_policies(table):
            op.execute(stmt)
