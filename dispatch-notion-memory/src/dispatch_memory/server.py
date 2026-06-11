"""MCP server exposing memory tools for Claude Desktop / Cowork / Dispatch.

Run with:
  stdio:  python -m dispatch_memory.server
  http:   python -m dispatch_memory.server --transport http --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from mcp.server import Server
from mcp.types import TextContent, Tool

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = str(_PACKAGE_ROOT / "config" / "config.yaml")

from .consolidation import Consolidator
from .entities import EntityExtractor
from .models import (
    BootstrapPayload, Memory, MemoryStatus, MemoryType, TTLClass,
)
from .storage import EmbeddingCache, NotionStore

logger = logging.getLogger(__name__)

# ── Tool JSON Schemas ─────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    Tool(
        name="memory_store",
        description=(
            "Store a new memory. Extracts entities via NER, generates an embedding, "
            "and saves to both Notion and the local vector cache. Use for facts, "
            "preferences, project context, standing instructions, etc."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The memory content to store.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["preference", "project", "person", "decision",
                             "workflow", "fact", "interaction", "standing_instruction", "agentic"],
                    "default": "fact",
                    "description": "Classification of the memory.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for categorisation.",
                },
                "significance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Importance score (0-1). Defaults based on memory_type.",
                },
                "ttl_class": {
                    "type": "string",
                    "enum": ["permanent", "long", "medium", "short"],
                    "default": "medium",
                    "description": "Decay half-life class.",
                },
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="memory_search",
        description=(
            "Semantic search across stored memories using local vector embeddings. "
            "Returns the most relevant memories ranked by cosine similarity. "
            "Use when you need to find memories related to a topic or question."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query.",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Max results to return.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["preference", "project", "person", "decision",
                             "workflow", "fact", "interaction", "standing_instruction", "agentic"],
                    "description": "Filter by memory type.",
                },
                "min_significance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum significance threshold.",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memory_retrieve",
        description=(
            "Fetch memories from Notion by page ID or by type/database. "
            "Use when you have a specific Notion page ID or want to list "
            "memories of a given type."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "notion_page_id": {
                    "type": "string",
                    "description": "Specific Notion page ID to retrieve.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["preference", "project", "person", "decision",
                             "workflow", "fact", "interaction", "standing_instruction", "agentic"],
                    "description": "Filter by memory type.",
                },
                "database": {
                    "type": "string",
                    "enum": ["inbox", "projects", "resources", "archive"],
                    "default": "projects",
                    "description": "Which PARA database to query.",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Max results.",
                },
            },
        },
    ),
    Tool(
        name="memory_update",
        description=(
            "Update an existing memory's content, tags, or significance. "
            "Re-extracts entities and re-indexes if content changes."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "notion_page_id": {
                    "type": "string",
                    "description": "Notion page ID of the memory to update.",
                },
                "content": {
                    "type": "string",
                    "description": "New content (triggers re-extraction and re-embedding).",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Replacement tags list.",
                },
                "significance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "New significance score.",
                },
            },
            "required": ["notion_page_id"],
        },
    ),
    Tool(
        name="memory_delete",
        description="Soft-delete a memory by archiving the Notion page and removing from local cache.",
        inputSchema={
            "type": "object",
            "properties": {
                "notion_page_id": {
                    "type": "string",
                    "description": "Notion page ID of the memory to delete.",
                },
            },
            "required": ["notion_page_id"],
        },
    ),
    Tool(
        name="memory_consolidate",
        description=(
            "Run a dream-inspired consolidation cycle: decay scoring, association "
            "discovery, compression, and archival. Modes: 'light' (decay only), "
            "'full' (decay + associations + compression), 'deep' (full + entity dedup)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["light", "full", "deep"],
                    "default": "full",
                    "description": "Consolidation depth.",
                },
            },
        },
    ),
    Tool(
        name="memory_recall",
        description=(
            "Compound query combining semantic search with structured Notion filters. "
            "Can filter by type, entities, tags, and significance simultaneously. "
            "Use for complex recall queries."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "memory_type": {
                    "type": "string",
                    "enum": ["preference", "project", "person", "decision",
                             "workflow", "fact", "interaction", "standing_instruction", "agentic"],
                    "description": "Filter by memory type.",
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by entity names (e.g. person or org names).",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by tags.",
                },
                "min_significance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Minimum significance threshold.",
                },
                "query": {
                    "type": "string",
                    "description": "Optional semantic search query to combine with filters.",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Max results.",
                },
            },
        },
    ),
    Tool(
        name="memory_refresh",
        description=(
            "Bump a memory's access metrics without modifying content. "
            "Use when a memory was referenced or relevant in a conversation."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "notion_page_id": {
                    "type": "string",
                    "description": "Notion page ID of the memory to refresh.",
                },
            },
            "required": ["notion_page_id"],
        },
    ),
    Tool(
        name="memory_compress",
        description=(
            "Run a significance-aware decay pass. Archives memories that have "
            "fallen below the archive threshold and marks forgotten memories."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="memory_status",
        description="Return memory statistics across all Notion databases and the local cache.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="memory_bootstrap",
        description=(
            "Return a curated context payload for session initialization. "
            "Includes standing instructions, active projects, and recent "
            "high-significance memories. Call at the start of a session."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
]


class MemoryService:
    """Core memory service orchestrating Notion, embeddings, entities, and consolidation."""

    def __init__(self, config_path: str = _DEFAULT_CONFIG):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        notion_cfg = self.config["notion"]
        self.notion = NotionStore(
            token=notion_cfg["token"],
            database_ids=notion_cfg["databases"],
        )

        storage_cfg = self.config["storage"]
        self.cache = EmbeddingCache(
            db_path=storage_cfg["sqlite_path"],
            model_name=storage_cfg["embedding_model"],
            embedding_dim=storage_cfg["embedding_dim"],
        )

        entity_cfg = self.config.get("entities", {})
        self.extractor = EntityExtractor(
            model_name=entity_cfg.get("spacy_model", "en_core_web_sm")
        )
        self.custom_entities = entity_cfg.get("custom_entities", [])

        sig_cfg = self.config.get("significance", {})
        self.consolidator = Consolidator(
            archive_threshold=sig_cfg.get("archive_threshold", 0.2),
            forget_threshold=sig_cfg.get("forget_threshold", 0.1),
        )

        self.significance_defaults = sig_cfg.get("defaults", {})
        self.half_lives = sig_cfg.get("half_lives", {})

    # ── Phase 1: Core Operations ──────────────────────────────────────

    async def memory_store(
        self,
        content: str,
        memory_type: str = "fact",
        tags: Optional[list[str]] = None,
        significance: Optional[float] = None,
        ttl_class: str = "medium",
    ) -> dict:
        """Create a memory: extract entities, embed, store in Notion + local cache."""
        mtype = MemoryType(memory_type)

        # Determine significance
        if significance is None:
            significance = self.significance_defaults.get(memory_type, 0.5)

        # Extract entities (spaCy + Coalition custom patterns)
        entities = self.extractor.extract_with_custom_entities(
            content, custom_entities=self.custom_entities
        )

        # Build memory
        memory = Memory(
            content=content,
            memory_type=mtype,
            tags=tags or [],
            entities=entities,
            significance=significance,
            ttl_class=TTLClass(ttl_class),
        )

        # Store in Notion
        memory = await self.notion.store(memory)

        # Index locally
        self.cache.index_memory(
            memory_id=memory.id,
            content=content,
            notion_page_id=memory.notion_page_id,
            content_hash=memory.content_hash,
            memory_type=memory_type,
            significance=significance,
            tags=tags or [],
            entities=[e.name for e in entities],
        )

        return {
            "id": memory.id,
            "notion_page_id": memory.notion_page_id,
            "entities_extracted": len(entities),
            "entity_names": [e.name for e in entities],
            "significance": significance,
            "ttl_class": ttl_class,
        }

    async def memory_search(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        min_significance: Optional[float] = None,
    ) -> list[dict]:
        """Semantic search via local embedding cache."""
        results = self.cache.search(
            query=query,
            limit=limit,
            memory_type=memory_type,
            min_significance=min_significance,
        )

        # Hydrate from Notion for top results
        hydrated = []
        for r in results:
            if r.get("notion_page_id"):
                memory = await self.notion.retrieve(r["notion_page_id"])
                if memory:
                    hydrated.append({
                        "content": memory.content,
                        "memory_type": memory.memory_type.value,
                        "tags": memory.tags,
                        "entities": memory.entity_names,
                        "significance": memory.significance,
                        "similarity_score": r["score"],
                        "notion_page_id": r["notion_page_id"],
                    })
            else:
                hydrated.append(r)

        return hydrated

    async def memory_retrieve(
        self,
        notion_page_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        database: str = "projects",
        limit: int = 20,
    ) -> list[dict]:
        """Fetch memories by ID or type from Notion."""
        if notion_page_id:
            memory = await self.notion.retrieve(notion_page_id)
            if memory:
                return [memory.model_dump(exclude={"embedding"})]
            return []

        mtype = MemoryType(memory_type) if memory_type else None
        memories = await self.notion.query_database(
            database=database,
            memory_type=mtype,
            limit=limit,
        )
        return [m.model_dump(exclude={"embedding"}) for m in memories]

    async def memory_update(
        self,
        notion_page_id: str,
        content: Optional[str] = None,
        tags: Optional[list[str]] = None,
        significance: Optional[float] = None,
    ) -> dict:
        """Update a memory's content or metadata."""
        memory = await self.notion.retrieve(notion_page_id)
        if not memory:
            return {"error": f"Memory not found: {notion_page_id}"}

        if content:
            memory.content = content
            memory.entities = self.extractor.extract(content)
        if tags is not None:
            memory.tags = tags
        if significance is not None:
            memory.significance = significance

        memory = await self.notion.update(memory)

        # Re-index locally
        if content:
            self.cache.index_memory(
                memory_id=memory.id,
                content=memory.content,
                notion_page_id=memory.notion_page_id,
                content_hash=memory.content_hash,
                memory_type=memory.memory_type.value,
                significance=memory.significance,
                tags=memory.tags,
                entities=[e.name for e in memory.entities],
            )

        return {"updated": True, "notion_page_id": notion_page_id}

    async def memory_delete(self, notion_page_id: str) -> dict:
        """Delete a memory."""
        memory = await self.notion.retrieve(notion_page_id)
        if memory:
            await self.notion.delete(memory)
            self.cache.remove(memory.id)
            return {"deleted": True}
        return {"error": "Memory not found"}

    async def memory_consolidate(self, mode: str = "full") -> dict:
        """Run dream consolidation cycle."""
        # Pull all active memories from Notion
        memories = await self.notion.query_all_active()

        # Run consolidation
        summary = self.consolidator.consolidate(memories, mode=mode)

        # Apply archival actions to Notion
        decay_result = self.consolidator.run_decay_pass(memories)
        for memory in decay_result["archive"]:
            await self.notion.archive(memory)
            self.cache.update_status(memory.id, "archived")

        for memory in decay_result["forget"]:
            self.cache.update_status(memory.id, "forgotten")

        return summary

    # ── Phase 2: Enhanced Operations ──────────────────────────────────

    async def memory_recall(
        self,
        memory_type: Optional[str] = None,
        entities: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        min_significance: Optional[float] = None,
        query: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Compound query across type, entities, significance, and semantic search."""
        results = []

        # If semantic query provided, start with embedding search
        if query:
            results = await self.memory_search(
                query=query,
                limit=limit * 2,  # Over-fetch to allow filtering
                memory_type=memory_type,
                min_significance=min_significance,
            )

        # Also query Notion for structured filters
        for db_name in ["projects", "resources", "inbox"]:
            mtype = MemoryType(memory_type) if memory_type else None
            notion_results = await self.notion.query_database(
                database=db_name,
                memory_type=mtype,
                tags=tags,
                min_significance=min_significance,
                limit=limit,
            )

            for memory in notion_results:
                # Filter by entity if requested
                if entities:
                    memory_entity_names = {e.name_lower for e in memory.entities}
                    if not any(e.lower() in memory_entity_names for e in entities):
                        continue

                entry = memory.model_dump(exclude={"embedding"})
                # Avoid duplicates
                if not any(
                    r.get("notion_page_id") == memory.notion_page_id
                    for r in results
                ):
                    results.append(entry)

        return results[:limit]

    async def memory_refresh(self, notion_page_id: str) -> dict:
        """Bump access metrics without modifying content."""
        memory = await self.notion.retrieve(notion_page_id)
        if not memory:
            return {"error": "Memory not found"}

        memory.refresh()
        await self.notion.update(memory)
        return {"refreshed": True, "access_count": memory.access_count}

    async def memory_compress(self) -> dict:
        """Run significance-aware decay pass."""
        memories = await self.notion.query_all_active()
        result = self.consolidator.run_decay_pass(memories)

        # Apply state changes
        for memory in result["archive"]:
            await self.notion.archive(memory)
            self.cache.update_status(memory.id, "archived")

        for memory in result["forget"]:
            self.cache.update_status(memory.id, "forgotten")

        return {
            "active": len(result["active"]),
            "archived": len(result["archive"]),
            "forgotten": len(result["forget"]),
        }

    async def memory_status(self) -> dict:
        """Return memory stats across all databases."""
        notion_stats = await self.notion.get_stats()
        cache_stats = self.cache.get_stats()
        return {
            "notion": notion_stats,
            "local_cache": cache_stats,
        }

    async def memory_bootstrap(self) -> dict:
        """Return curated context payload for session initialization."""
        boot_cfg = self.config.get("bootstrap", {})
        threshold = boot_cfg.get("significance_threshold", 0.8)
        max_memories = boot_cfg.get("max_memories", 50)

        payload = BootstrapPayload()

        # Standing instructions
        if boot_cfg.get("include_standing_instructions", True):
            instructions = await self.notion.query_database(
                database="resources",
                min_significance=threshold,
                limit=max_memories,
            )
            payload.standing_instructions = [
                m for m in instructions
                if m.memory_type == MemoryType.STANDING_INSTRUCTION
                or m.significance >= 0.9
            ]

        # Active projects
        if boot_cfg.get("include_active_projects", True):
            projects = await self.notion.query_database(
                database="projects",
                limit=max_memories,
            )
            payload.active_projects = projects

        # Recent high-significance
        if boot_cfg.get("include_recent_high_significance", True):
            recent = await self.notion.query_all_active(
                min_significance=threshold,
                limit=max_memories,
            )
            payload.recent_high_significance = recent

        # Stats
        payload.stats = self.cache.get_stats()

        return payload.model_dump(exclude={"standing_instructions": {"__all__": {"embedding"}},
                                           "active_projects": {"__all__": {"embedding"}},
                                           "recent_high_significance": {"__all__": {"embedding"}}})


# ── MCP Server Wiring ─────────────────────────────────────────────────

def create_mcp_server(config_path: str = _DEFAULT_CONFIG) -> Server:
    """Create and configure the MCP Server with all memory tools registered."""

    server = Server("dispatch-notion-memory")

    # Lazy-init the service on first tool call (avoids blocking at import time
    # and lets the transport start before heavy model loading).
    _service: dict[str, Optional[MemoryService]] = {"instance": None}

    def _get_service() -> MemoryService:
        if _service["instance"] is None:
            logger.info("Initialising MemoryService (first tool call)…")
            _service["instance"] = MemoryService(config_path)
            logger.info("MemoryService ready.")
        return _service["instance"]

    # ── list_tools ────────────────────────────────────────────────────

    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        return TOOL_DEFINITIONS

    # ── call_tool ─────────────────────────────────────────────────────

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        svc = _get_service()

        try:
            # Dispatch to the matching MemoryService method
            handler = getattr(svc, name, None)
            if handler is None:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"}),
                )]

            result = await handler(**arguments)

            # Serialise – handle datetimes & enums via default
            return [TextContent(
                type="text",
                text=json.dumps(result, default=str),
            )]

        except Exception as exc:
            logger.exception(f"Tool {name} failed")
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(exc)}),
            )]

    return server


def load_config(config_path: str = _DEFAULT_CONFIG) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Entry Points ──────────────────────────────────────────────────────

def main():
    """Entry point for running the MCP server.

    Supports two transports:
      stdio (default) — for Claude Desktop / Cowork / Claude Code
      http            — SSE-based HTTP transport on configurable port
    """
    parser = argparse.ArgumentParser(
        description="Dispatch Notion Memory — MCP Server",
    )
    parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "http"],
        default=None,
        help="Transport layer (default: read from config, fallback stdio).",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port for HTTP transport (default: read from config, fallback 8765).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host for HTTP transport (default: read from config, fallback 127.0.0.1).",
    )
    parser.add_argument(
        "--config", "-c",
        default=_DEFAULT_CONFIG,
        help="Path to config YAML (default: resolved from package dir).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config for transport defaults
    config: dict[str, Any] = {}
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning(f"Config not found at {args.config} — using defaults.")

    server_cfg = config.get("server", {})
    transport = args.transport or server_cfg.get("transport", "stdio")
    host = args.host or server_cfg.get("host", "127.0.0.1")
    port = args.port or server_cfg.get("port", 8765)

    mcp_server = create_mcp_server(config_path=args.config)

    if transport == "stdio":
        logger.info("Starting MCP server on stdio transport…")
        _run_stdio(mcp_server)
    elif transport == "http":
        logger.info(f"Starting MCP server on http://{host}:{port}/sse …")
        _run_http(mcp_server, host, port)
    else:
        logger.error(f"Unknown transport: {transport}")
        sys.exit(1)


def _run_stdio(server: Server) -> None:
    """Run the server over stdin/stdout (Claude Desktop / Cowork integration)."""
    import signal
    from mcp.server.stdio import stdio_server

    async def _stdin_watchdog():
        """Exit if stdin closes (parent disconnected)."""
        loop = asyncio.get_event_loop()
        while True:
            await asyncio.sleep(30)
            if sys.stdin.closed:
                logger.info("stdin closed — parent disconnected, exiting.")
                os._exit(0)

    async def _main():
        asyncio.create_task(_stdin_watchdog())
        async with stdio_server() as (read_stream, write_stream):
            init_options = server.create_initialization_options()
            await server.run(read_stream, write_stream, init_options)
        logger.info("stdio_server context exited, shutting down.")

    asyncio.run(_main())


def _run_http(server: Server, host: str, port: int) -> None:
    """Run the server over SSE HTTP transport."""
    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            init_options = server.create_initialization_options()
            await server.run(read_stream, write_stream, init_options)

    app = Starlette(
        debug=True,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
