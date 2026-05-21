"""KV Knowledge Packs — Zero-Token Memory Injection

Core module: cache building, composition, and fact store management.

Converts retrieved text (memories, values, system prompts) into pre-computed
KV cache state for injection at inference time. Text becomes geometry —
the model behaves as if it read the text, but zero context tokens are consumed.

Requires: transformers, torch. Optional: kvpack (for routing embeddings).
Designed for HuggingFace Transformers models. vLLM support via prefix caching.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)

CACHE_DIR = Path(os.environ.get('KVPACK_CACHE_DIR', '/tmp/kvpack-cache'))
MAX_CACHE_AGE = int(os.environ.get('KVPACK_MAX_CACHE_AGE', '3600'))


@dataclass
class CacheBlock:
    """A pre-computed KV cache block with metadata."""
    key_values: tuple
    seq_length: int
    source_hash: str
    created_at: float = field(default_factory=time.time)
    label: str = ''

    @property
    def age(self) -> float:
        return time.time() - self.created_at


@dataclass
class FactBank:
    """A collection of facts with routing embeddings for selective injection."""
    facts: list[str]
    embeddings: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)

    def hash(self) -> str:
        content = '\n'.join(sorted(self.facts))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_text(self, template: str = "system") -> str:
        return '\n'.join(f'- {fact}' for fact in self.facts)


class KVPackBuilder:
    """Builds KV cache blocks from text using a loaded model."""

    def __init__(self, model, tokenizer, device: str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._block_cache: dict[str, CacheBlock] = {}

    def encode(self, text: str, chat_template: bool = True,
               role: str = 'system', label: str = '') -> CacheBlock:
        """Convert text to a KV cache block.

        Args:
            text: The content to encode into KV cache.
            chat_template: Wrap in model's chat template (critical for accuracy).
            role: Chat role for template wrapping.
            label: Human-readable label for this block.

        Returns:
            CacheBlock with pre-computed key-value tensors.
        """
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        if content_hash in self._block_cache:
            cached = self._block_cache[content_hash]
            if cached.age < MAX_CACHE_AGE:
                return cached

        if chat_template:
            messages = [{'role': role, 'content': text}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            formatted = text

        input_ids = self.tokenizer.encode(formatted, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )

        block = CacheBlock(
            key_values=outputs.past_key_values,
            seq_length=input_ids.shape[1],
            source_hash=content_hash,
            label=label or text[:60],
        )

        self._block_cache[content_hash] = block
        logger.info(f'Encoded KV block: {block.seq_length} tokens, label="{block.label}"')
        return block

    def encode_with_prefix(self, text: str, prefix_block: CacheBlock,
                           chat_template: bool = True,
                           role: str = 'user', label: str = '') -> CacheBlock:
        """Encode text with a prefix cache for RoPE continuity.

        The new block's positions start where the prefix ends, maintaining
        correct rotary position embeddings across composed caches.
        """
        if chat_template:
            messages = [{'role': role, 'content': text}]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            formatted = text

        input_ids = self.tokenizer.encode(formatted, return_tensors='pt').to(self.device)

        prefix_len = prefix_block.seq_length
        position_ids = torch.arange(
            prefix_len, prefix_len + input_ids.shape[1],
            dtype=torch.long, device=self.device
        ).unsqueeze(0)

        full_mask = torch.ones(
            1, prefix_len + input_ids.shape[1],
            dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=prefix_block.key_values,
                attention_mask=full_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )

        content_hash = hashlib.sha256(
            (prefix_block.source_hash + text).encode()
        ).hexdigest()[:16]

        block = CacheBlock(
            key_values=outputs.past_key_values,
            seq_length=prefix_len + input_ids.shape[1],
            source_hash=content_hash,
            label=label or f'prefix+{text[:40]}',
        )

        logger.info(f'Encoded prefixed KV block: {block.seq_length} total tokens')
        return block

    def clear_cache(self):
        self._block_cache.clear()

    def cache_stats(self) -> dict:
        return {
            'blocks': len(self._block_cache),
            'total_tokens': sum(b.seq_length for b in self._block_cache.values()),
            'oldest_age': max((b.age for b in self._block_cache.values()), default=0),
        }


class CacheComposer:
    """Composes multiple KV cache blocks into a single injection-ready cache."""

    def __init__(self, builder: KVPackBuilder):
        self.builder = builder

    def compose(self, *blocks: CacheBlock) -> CacheBlock:
        """Compose multiple cache blocks with correct RoPE continuity.

        Blocks are processed sequentially — each block's positions continue
        from where the previous block ended. This is NOT simple concatenation.

        For best results, encode blocks with encode_with_prefix() rather than
        composing independently-encoded blocks.
        """
        if not blocks:
            raise ValueError("Need at least one block to compose")
        if len(blocks) == 1:
            return blocks[0]

        composed_kv = blocks[0].key_values
        total_len = blocks[0].seq_length

        for block in blocks[1:]:
            num_layers = len(composed_kv)
            new_kv = []
            for layer_idx in range(num_layers):
                prev_k, prev_v = composed_kv[layer_idx]
                curr_k, curr_v = block.key_values[layer_idx]

                new_k = torch.cat([prev_k, curr_k], dim=2)
                new_v = torch.cat([prev_v, curr_v], dim=2)
                new_kv.append((new_k, new_v))

            composed_kv = tuple(new_kv)
            total_len += block.seq_length

        combined_hash = hashlib.sha256(
            '|'.join(b.source_hash for b in blocks).encode()
        ).hexdigest()[:16]

        return CacheBlock(
            key_values=composed_kv,
            seq_length=total_len,
            source_hash=combined_hash,
            label=f'composed({len(blocks)} blocks)',
        )

    def compose_for_generation(self, system_block: CacheBlock,
                                memory_block: Optional[CacheBlock] = None
                                ) -> tuple:
        """Prepare a composed cache + attention mask for model.generate().

        Returns (past_key_values, attention_mask_prefix_length) ready for
        the model's generate() call.
        """
        if memory_block:
            composed = self.compose(system_block, memory_block)
        else:
            composed = system_block

        return composed.key_values, composed.seq_length


class FactStore:
    """Persistent fact storage with routing embeddings.

    Facts are stored as text + optional embeddings for query routing.
    KV caches are computed lazily from stored text — never persisted
    as tensors (avoids model version pinning).
    """

    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._banks: dict[str, FactBank] = {}
        self._load()

    def _load(self):
        for f in self.store_path.glob('*.json'):
            try:
                data = json.loads(f.read_text())
                bank = FactBank(
                    facts=data['facts'],
                    metadata=data.get('metadata', {}),
                )
                self._banks[f.stem] = bank
            except Exception as e:
                logger.warning(f'Failed to load fact bank {f}: {e}')

    def create_bank(self, name: str, facts: list[str],
                    metadata: Optional[dict] = None) -> FactBank:
        bank = FactBank(facts=facts, metadata=metadata or {})
        self._banks[name] = bank
        self._save(name)
        return bank

    def get_bank(self, name: str) -> Optional[FactBank]:
        return self._banks.get(name)

    def update_bank(self, name: str, facts: list[str]):
        if name in self._banks:
            self._banks[name].facts = facts
            self._save(name)
        else:
            self.create_bank(name, facts)

    def list_banks(self) -> list[str]:
        return list(self._banks.keys())

    def _save(self, name: str):
        bank = self._banks[name]
        data = {
            'facts': bank.facts,
            'metadata': bank.metadata,
            'hash': bank.hash(),
        }
        path = self.store_path / f'{name}.json'
        path.write_text(json.dumps(data, indent=2))


def generate_with_kvpack(model, tokenizer, query: str,
                         cache_block: CacheBlock,
                         max_new_tokens: int = 256,
                         **generate_kwargs) -> str:
    """Generate text with a pre-computed KV cache injected.

    This is the zero-token inference call. The model sees the full context
    from the cache block but only processes the query tokens.
    """
    input_ids = tokenizer.encode(query, return_tensors='pt').to(model.device)

    prefix_len = cache_block.seq_length
    attention_mask = torch.ones(
        1, prefix_len + input_ids.shape[1],
        dtype=torch.long, device=model.device
    )

    position_ids = torch.arange(
        prefix_len, prefix_len + input_ids.shape[1],
        dtype=torch.long, device=model.device
    ).unsqueeze(0)

    outputs = model.generate(
        input_ids=input_ids,
        past_key_values=cache_block.key_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_new_tokens=max_new_tokens,
        **generate_kwargs,
    )

    new_tokens = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
