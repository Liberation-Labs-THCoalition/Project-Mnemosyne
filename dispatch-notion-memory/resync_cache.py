"""Resync local embedding cache from Notion.

Pulls all active memories from Notion and re-indexes them in the local
sqlite-vec cache. Fixes cache↔Notion desync (handoff bug #2).
"""

import asyncio
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[resync] %(message)s")
log = logging.getLogger(__name__)


async def resync():
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    from src.dispatch_memory.storage.notion_store import NotionStore
    from src.dispatch_memory.storage.embedding_cache import EmbeddingCache
    from src.dispatch_memory.entities.extractor import EntityExtractor

    notion_cfg = config["notion"]
    storage_cfg = config["storage"]
    entity_cfg = config.get("entities", {})

    notion = NotionStore(
        token=notion_cfg["token"],
        database_ids=notion_cfg["databases"],
    )

    cache = EmbeddingCache(
        db_path=storage_cfg["sqlite_path"],
        model_name=storage_cfg["embedding_model"],
        embedding_dim=storage_cfg["embedding_dim"],
    )

    extractor = EntityExtractor(
        model_name=entity_cfg.get("spacy_model", "en_core_web_sm")
    )
    custom_entities = entity_cfg.get("custom_entities", [])

    log.info("Pulling all active memories from Notion...")
    memories = await notion.query_all_active()
    log.info(f"Found {len(memories)} active memories in Notion")

    existing_stats = cache.get_stats()
    log.info(f"Local cache before: {existing_stats}")

    indexed = 0
    errors = 0
    for mem in memories:
        try:
            entities = extractor.extract_with_custom_entities(
                mem.content, custom_entities
            )
            mem.entities = entities

            cache.index_memory(
                memory_id=mem.id or mem.content[:32],
                content=mem.content,
                memory_type=mem.memory_type.value if mem.memory_type else "fact",
                tags=mem.tags,
                entities=[e.name for e in entities],
                significance=mem.significance or 0.5,
            )
            indexed += 1
            if indexed % 10 == 0:
                log.info(f"  indexed {indexed}/{len(memories)}")
        except Exception as e:
            errors += 1
            log.warning(f"  Failed to index: {e}")

    after_stats = cache.get_stats()
    log.info(f"Local cache after: {after_stats}")
    log.info(f"Resync complete: {indexed} indexed, {errors} errors")


if __name__ == "__main__":
    asyncio.run(resync())
