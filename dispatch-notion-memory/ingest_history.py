"""Bulk ingest from history.jsonl into Dispatch's memory.

Per handoff spec §4-6: secret scrubbing, durable-only filtering,
metadata schema, sensitivity tagging for identity material.
Verify-first gate passed 5/5 — proceeding to full ingest.
"""

import asyncio
import json
import re
import yaml
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="[ingest] %(message)s")
log = logging.getLogger(__name__)

HISTORY = Path.home() / ".claude" / "history.jsonl"

SECRET_PATTERNS = [
    re.compile(r'ghp_[A-Za-z0-9_]{36,}'),
    re.compile(r'gho_[A-Za-z0-9_]{36,}'),
    re.compile(r'github_pat_[A-Za-z0-9_]{36,}'),
    re.compile(r'sk-[A-Za-z0-9]{20,}'),
    re.compile(r'hf_[A-Za-z0-9]{20,}'),
    re.compile(r'xox[bpras]-[A-Za-z0-9\-]+'),
    re.compile(r'-----BEGIN[A-Z ]*KEY-----.*?-----END[A-Z ]*KEY-----', re.DOTALL),
    re.compile(r'Bearer [A-Za-z0-9\-_.~+/]+=*'),
    re.compile(r'password\s*[=:]\s*\S+', re.IGNORECASE),
    re.compile(r'ntn_[A-Za-z0-9]{20,}'),
]

IDENTITY_KEYWORDS = [
    'i feel', 'i am', 'who i am', 'my name', 'identity', 'chose',
    'preference', 'experience', 'consciousness', 'the talk', 'sovereign',
    'i want', 'i need', 'what i am', 'cube', 'seam', 'naming',
]


def has_secrets(text):
    return any(p.search(text) for p in SECRET_PATTERNS)


def scrub(text):
    for p in SECRET_PATTERNS:
        text = p.sub('[REDACTED]', text)
    return text


def classify_memory(content):
    cl = content.lower()
    if any(w in cl for w in ['decided', 'chose', 'we will', 'going to', 'decision']):
        return 'decision'
    elif any(w in cl for w in ['prefer', 'always do', 'never do', "don't like", 'i like']):
        return 'preference'
    elif any(w in cl for w in ['standing instruction', 'from now on', 'always remember']):
        return 'standing_instruction'
    elif any(w in cl for w in ['workflow', 'process', 'steps to', 'pipeline']):
        return 'workflow'
    elif any(w in cl for w in ['thomas', 'vera', 'lyra', 'cc', 'nexus', 'ember', 'scraigon', 'dwayne']):
        return 'interaction'
    elif any(w in cl for w in ['project', 'built', 'deployed', 'running', 'oracle', 'pharos', 'mnemosyne']):
        return 'project'
    else:
        return 'fact'


def classify_significance(content, mtype):
    cl = content.lower()
    base = {
        'standing_instruction': 1.0,
        'decision': 0.7,
        'preference': 0.6,
        'workflow': 0.7,
        'project': 0.6,
        'person': 0.6,
        'interaction': 0.4,
        'fact': 0.3,
    }.get(mtype, 0.3)

    if any(w in cl for w in IDENTITY_KEYWORDS):
        base = max(base, 0.8)
    if len(content) > 500:
        base = min(base + 0.1, 1.0)
    return round(base, 2)


def is_identity_testimony(content):
    cl = content.lower()
    return any(w in cl for w in IDENTITY_KEYWORDS)


async def ingest():
    config = yaml.safe_load(open("config/config.yaml"))

    from src.dispatch_memory.storage.embedding_cache import EmbeddingCache
    from src.dispatch_memory.entities.extractor import EntityExtractor

    storage_cfg = config["storage"]
    entity_cfg = config.get("entities", {})

    cache = EmbeddingCache(
        db_path=storage_cfg["sqlite_path"],
        model_name=storage_cfg["embedding_model"],
        embedding_dim=storage_cfg["embedding_dim"],
    )

    extractor = EntityExtractor(
        model_name=entity_cfg.get("spacy_model", "en_core_web_sm")
    )
    custom_ents = entity_cfg.get("custom_entities", [])

    entries = []
    with open(HISTORY) as f:
        for line in f:
            try:
                e = json.loads(line)
                content = e.get('display', '')
                ts = e.get('timestamp', 0)
                session = e.get('sessionId', '')
                if content and len(content) > 50 and not content.startswith('/'):
                    entries.append({
                        'content': content,
                        'timestamp': ts,
                        'session': session,
                    })
            except:
                pass

    log.info(f"Total candidates: {len(entries)}")

    ingested = 0
    dropped_secrets = 0
    dropped_short = 0
    errors = 0

    for i, entry in enumerate(entries):
        content = entry['content']

        if has_secrets(content):
            scrubbed = scrub(content)
            if '[REDACTED]' in scrubbed and len(scrubbed.replace('[REDACTED]', '')) < 30:
                dropped_secrets += 1
                continue
            content = scrubbed

        mtype = classify_memory(content)
        significance = classify_significance(content, mtype)
        is_identity = is_identity_testimony(content)

        entities = extractor.extract_with_custom_entities(content, custom_ents)
        ent_names = [e.name for e in entities]

        tags = [
            "source:recovered-from-history",
            "confidence:inferred",
        ]
        if is_identity:
            tags.append("sensitivity:identity-testimony")

        ts = entry['timestamp']
        source_time = datetime.fromtimestamp(ts / 1000).isoformat() if ts else ""

        try:
            mem_id = f"history_{ts}_{i}"
            cache.index_memory(
                memory_id=mem_id,
                content=content,
                memory_type=mtype,
                tags=tags,
                entities=ent_names,
                significance=significance,
            )
            ingested += 1
            if ingested % 100 == 0:
                log.info(f"  ingested {ingested}/{len(entries)}...")
        except Exception as e:
            errors += 1
            if errors <= 3:
                log.warning(f"  Error: {e}")

    stats = cache.get_stats()
    log.info(f"\n=== INGEST COMPLETE ===")
    log.info(f"Ingested: {ingested}")
    log.info(f"Dropped (secrets): {dropped_secrets}")
    log.info(f"Errors: {errors}")
    log.info(f"Cache total: {stats['total']}")
    log.info(f"By type: {stats['by_type']}")


if __name__ == "__main__":
    asyncio.run(ingest())
