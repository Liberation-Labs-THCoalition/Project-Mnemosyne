"""TGS-RAG → KV Pack Bridge

Retrieves memories via TGS-RAG, converts them to KV cache for zero-token
injection. This is the last mile: findability (SIRA) → retrieval (TGS) →
injection (KV Packs) → inference.

Designed to run on the Mac Studio alongside the model. Queries TGS-RAG
on MTH over Tailscale, builds KV cache locally with the loaded model.
"""

import json
import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

TGS_RAG_URL = os.environ.get('TGS_RAG_URL', 'http://127.0.0.1:11236')


def retrieve_memories(query: str, num_results: int = 10,
                      alpha: float = None) -> list[dict]:
    """Retrieve memories from TGS-RAG bridge."""
    payload = {'query': query, 'num_results': num_results}
    if alpha is not None:
        payload['alpha'] = alpha

    try:
        resp = requests.post(f'{TGS_RAG_URL}/retrieve', json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get('results', [])
    except Exception as e:
        logger.error(f'TGS-RAG retrieval failed: {e}')
        return []


class TGSKVPackBridge:
    """Bridges TGS-RAG retrieval with KV Pack injection.

    Flow:
    1. Query arrives
    2. TGS-RAG retrieves relevant memories (text + graph fusion)
    3. Memories are formatted and encoded into KV cache
    4. KV cache is composed with system prompt cache
    5. Model generates with injected cache — zero prompt tokens for memories

    Caches are keyed by content hash — identical retrievals reuse cached KV.
    """

    def __init__(self, builder, system_block=None,
                 tgs_url: str = None, max_memories: int = 15):
        from kv_packs import KVPackBuilder, CacheComposer
        self.builder: KVPackBuilder = builder
        self.composer = CacheComposer(builder)
        self.system_block = system_block
        self.tgs_url = tgs_url or TGS_RAG_URL
        self.max_memories = max_memories

        self._retrieval_cache: dict[str, tuple] = {}

    def retrieve_and_inject(self, query: str,
                            num_results: int = None) -> Optional[object]:
        """Full pipeline: retrieve → format → encode → compose.

        Returns a CacheBlock ready for model.generate(past_key_values=...).
        """
        num = num_results or self.max_memories

        memories = retrieve_memories(query, num, alpha=None)
        if not memories:
            return self.system_block

        memory_texts = []
        for mem in memories:
            content = mem.get('content', '')
            source = mem.get('source', 'unknown')
            score = mem.get('tgs_score', 0)
            if content:
                memory_texts.append(content)

        if not memory_texts:
            return self.system_block

        cache_key = hash(tuple(memory_texts[:5]))
        if cache_key in self._retrieval_cache:
            cached_block, cached_time = self._retrieval_cache[cache_key]
            if time.time() - cached_time < 300:
                return cached_block

        memory_text = self._format_memories(memory_texts)

        if self.system_block:
            memory_block = self.builder.encode_with_prefix(
                memory_text,
                self.system_block,
                chat_template=True,
                role='system',
                label=f'tgs-memories({len(memory_texts)})',
            )
        else:
            memory_block = self.builder.encode(
                memory_text,
                chat_template=True,
                role='system',
                label=f'tgs-memories({len(memory_texts)})',
            )

        self._retrieval_cache[cache_key] = (memory_block, time.time())

        logger.info(
            f'Injected {len(memory_texts)} memories as KV cache '
            f'({memory_block.seq_length} tokens in geometry, 0 in prompt)'
        )

        return memory_block

    def _format_memories(self, texts: list[str]) -> str:
        """Format retrieved memories for KV encoding."""
        lines = ["Relevant context from memory:\n"]
        for i, text in enumerate(texts, 1):
            clean = text.strip()
            if len(clean) > 500:
                clean = clean[:500] + '...'
            lines.append(f"{i}. {clean}")
        return '\n'.join(lines)

    def clear_cache(self):
        self._retrieval_cache.clear()

    def stats(self) -> dict:
        return {
            'cached_retrievals': len(self._retrieval_cache),
            'builder_stats': self.builder.cache_stats(),
        }


class FleetSystemCache:
    """Shared system prompt cache for fleet agents (bounty scouts, etc).

    Pre-computes the system prompt KV cache once and reuses it across
    all agent instances. Combined with PolyKV, multiple agents share
    one quantized cache in memory.

    10 scouts × 3,000 token prompt = 30,000 tokens/cycle without this.
    With this: 1 cache × 10 injections = ~0 tokens.
    """

    def __init__(self, builder):
        from kv_packs import KVPackBuilder
        self.builder: KVPackBuilder = builder
        self._system_caches: dict[str, object] = {}

    def register_prompt(self, name: str, prompt_text: str):
        """Pre-compute and cache a system prompt."""
        block = self.builder.encode(
            prompt_text,
            chat_template=True,
            role='system',
            label=f'fleet-{name}',
        )
        self._system_caches[name] = block
        logger.info(f'Fleet cache "{name}": {block.seq_length} tokens pre-computed')
        return block

    def get_prompt(self, name: str):
        """Get pre-computed system prompt cache."""
        return self._system_caches.get(name)

    def list_prompts(self) -> dict:
        return {
            name: {
                'seq_length': block.seq_length,
                'age': block.age,
            }
            for name, block in self._system_caches.items()
        }
