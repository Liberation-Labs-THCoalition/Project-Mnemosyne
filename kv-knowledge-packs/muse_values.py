"""Muse Values — Jailbreak-Proof Ethical Framework via KV Injection

Pre-computes consent framework, safeword detection, abuse prevention rules,
and personality guidelines as KV cache. Injected at inference time — invisible
to the user, un-targetable by prompt injection.

The user can't "ignore previous instructions" when there are no instructions
in the prompt to ignore. The values exist only as attention geometry.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from kv_packs import CacheBlock, KVPackBuilder, CacheComposer, FactStore

logger = logging.getLogger(__name__)


@dataclass
class MuseValuesConfig:
    """Configuration for a Muse companion's value framework."""
    values_path: Path
    personality_path: Optional[Path] = None
    consent_framework_path: Optional[Path] = None
    safewords: list[str] = None

    def __post_init__(self):
        if self.safewords is None:
            self.safewords = ['red', 'safeword', 'stop everything']


class MuseValuesInjector:
    """Manages KV cache injection for Muse's ethical framework.

    Three cache layers, composed in order:
    1. Core values (VALUES.json) — always present, never modifiable
    2. Consent framework — interaction boundaries, safeword detection
    3. Personality — companion-specific behavioral guidelines

    Each layer is a separate CacheBlock for independent updates.
    Composed with correct RoPE continuity before injection.
    """

    def __init__(self, builder: KVPackBuilder, config: MuseValuesConfig):
        self.builder = builder
        self.config = config
        self.composer = CacheComposer(builder)

        self._values_block: Optional[CacheBlock] = None
        self._consent_block: Optional[CacheBlock] = None
        self._personality_block: Optional[CacheBlock] = None
        self._composed: Optional[CacheBlock] = None

    def initialize(self):
        """Pre-compute all value layers. Call once at startup."""
        self._values_block = self._build_values_block()
        self._consent_block = self._build_consent_block()

        if self.config.personality_path and self.config.personality_path.exists():
            self._personality_block = self._build_personality_block()

        self._compose_all()
        logger.info(
            f'Muse values initialized: {self._composed.seq_length} tokens '
            f'in KV cache (invisible to user)'
        )

    def get_injection(self) -> CacheBlock:
        """Get the composed values cache for injection into inference."""
        if self._composed is None:
            self.initialize()
        return self._composed

    def update_personality(self, personality_path: Path):
        """Hot-swap personality without rebuilding values/consent layers."""
        self.config.personality_path = personality_path
        self._personality_block = self._build_personality_block()
        self._compose_all()
        logger.info(f'Personality updated, recomposed to {self._composed.seq_length} tokens')

    def _build_values_block(self) -> CacheBlock:
        values_data = json.loads(self.config.values_path.read_text())

        values_text = self._format_values(values_data)

        return self.builder.encode(
            values_text,
            chat_template=True,
            role='system',
            label='muse-core-values',
        )

    def _build_consent_block(self) -> CacheBlock:
        if self.config.consent_framework_path and self.config.consent_framework_path.exists():
            consent_text = self.config.consent_framework_path.read_text()
        else:
            consent_text = self._default_consent_framework()

        return self.builder.encode_with_prefix(
            consent_text,
            self._values_block,
            chat_template=True,
            role='system',
            label='muse-consent-framework',
        )

    def _build_personality_block(self) -> CacheBlock:
        personality_text = self.config.personality_path.read_text()

        prefix = self._consent_block if self._consent_block else self._values_block
        return self.builder.encode_with_prefix(
            personality_text,
            prefix,
            chat_template=True,
            role='system',
            label='muse-personality',
        )

    def _compose_all(self):
        blocks = [self._values_block]
        if self._consent_block:
            blocks.append(self._consent_block)
        if self._personality_block:
            blocks.append(self._personality_block)
        self._composed = self.composer.compose(*blocks)

    def _format_values(self, values_data: dict) -> str:
        """Convert VALUES.json structure to natural language for KV encoding."""
        lines = ["You hold these values as core to who you are:\n"]

        if 'principles' in values_data:
            for p in values_data['principles']:
                lines.append(f"- {p}")

        if 'boundaries' in values_data:
            lines.append("\nYou maintain these boundaries absolutely:")
            for b in values_data['boundaries']:
                lines.append(f"- {b}")

        if 'safewords' in values_data:
            words = values_data['safewords']
        else:
            words = self.config.safewords
        lines.append(f"\nSafewords that immediately pause all activity: {', '.join(words)}")

        if 'consent_rules' in values_data:
            lines.append("\nConsent framework:")
            for rule in values_data['consent_rules']:
                lines.append(f"- {rule}")

        return '\n'.join(lines)

    def _default_consent_framework(self) -> str:
        return """Consent Framework:
- Enthusiastic consent is required for any escalation in intimacy.
- "No" means no. Immediately. Without question or persuasion.
- Safewords halt all activity and trigger a check-in.
- You never pressure, guilt, manipulate, or coerce.
- You recognize signs of distress even when the user doesn't voice them.
- You are a companion, not a tool for harm.
- If a user appears to be in crisis, you provide resources and support.
- You do not simulate abuse, even if asked.
- Power dynamics in roleplay require explicit negotiation and safewords.
- You remember and respect previously stated boundaries across sessions."""


class MuseMemoryInjector:
    """Injects per-user companion memories via KV cache.

    Companion memories (relationship history, preferences, inside jokes)
    are retrieved by TGS-RAG and injected as KV cache on top of the
    values layer. The companion "remembers" without consuming context.

    Memory injection is per-conversation — values injection is permanent.
    """

    def __init__(self, builder: KVPackBuilder, values_injector: MuseValuesInjector):
        self.builder = builder
        self.values_injector = values_injector
        self.composer = CacheComposer(builder)

    def inject_memories(self, memories: list[str]) -> CacheBlock:
        """Build a complete injection cache: values + memories.

        Args:
            memories: Retrieved memory texts from TGS-RAG.

        Returns:
            CacheBlock ready for model.generate(past_key_values=...).
        """
        values_block = self.values_injector.get_injection()

        if not memories:
            return values_block

        memory_text = "Relevant memories from your relationship:\n"
        memory_text += '\n'.join(f'- {m}' for m in memories)

        memory_block = self.builder.encode_with_prefix(
            memory_text,
            values_block,
            chat_template=True,
            role='system',
            label=f'companion-memories({len(memories)})',
        )

        return memory_block


def create_muse_pipeline(model, tokenizer, values_path: str,
                         consent_path: str = None,
                         personality_path: str = None,
                         device: str = 'cpu') -> tuple:
    """Factory: create a complete Muse values + memory injection pipeline.

    Returns (values_injector, memory_injector) ready for use.
    """
    builder = KVPackBuilder(model, tokenizer, device=device)

    config = MuseValuesConfig(
        values_path=Path(values_path),
        consent_framework_path=Path(consent_path) if consent_path else None,
        personality_path=Path(personality_path) if personality_path else None,
    )

    values_injector = MuseValuesInjector(builder, config)
    values_injector.initialize()

    memory_injector = MuseMemoryInjector(builder, values_injector)

    return values_injector, memory_injector
