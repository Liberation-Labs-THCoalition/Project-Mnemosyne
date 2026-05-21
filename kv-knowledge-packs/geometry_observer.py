"""Geometry Observer — Watch what KV injection does to attention patterns.

When we inject memories or values via KV cache, the model's attention geometry
changes. This module records those changes for Oracle's persistence layer —
tracking how injected knowledge shapes the model's internal state.

Integration point: Oracle persistence (oracle-memory) records KV-cache geometry
per layer per head. This module adds injection-aware annotations: what was
injected, which layers responded most, whether the injected content produced
coherent or diffuse attention patterns.

This is the observability layer for zero-token injection.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class GeometryReading:
    """A snapshot of attention geometry after KV injection."""
    timestamp: float
    injection_label: str
    injection_tokens: int
    query_tokens: int
    per_layer_entropy: list[float]
    per_layer_max_attention: list[float]
    injection_attention_mass: list[float]
    coherence_score: float
    metadata: dict = field(default_factory=dict)


class GeometryObserver:
    """Records how KV cache injection affects attention geometry.

    Wraps model forward passes to capture attention weights when generating
    with injected KV caches. Reports which layers attend most to injected
    content vs. the query, and whether attention is focused or diffuse.

    Feeds into Oracle's persistence layer for cross-session trajectory tracking.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.readings: list[GeometryReading] = []

    def observe_injection(self, query: str, cache_block,
                          label: str = '') -> GeometryReading:
        """Run a forward pass with injected cache and record attention geometry.

        Returns a GeometryReading with per-layer attention statistics.
        """
        input_ids = self.tokenizer.encode(query, return_tensors='pt').to(self.model.device)
        prefix_len = cache_block.seq_length

        attention_mask = torch.ones(
            1, prefix_len + input_ids.shape[1],
            dtype=torch.long, device=self.model.device
        )

        position_ids = torch.arange(
            prefix_len, prefix_len + input_ids.shape[1],
            dtype=torch.long, device=self.model.device
        ).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=cache_block.key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=True,
                return_dict=True,
            )

        attentions = outputs.attentions

        per_layer_entropy = []
        per_layer_max = []
        injection_mass = []

        for layer_attn in attentions:
            # layer_attn: [batch, heads, query_len, kv_len]
            avg_attn = layer_attn.mean(dim=1).squeeze(0)  # [query_len, kv_len]

            probs = avg_attn[-1]  # last query token's attention distribution
            entropy = -(probs * (probs + 1e-10).log()).sum().item()
            per_layer_entropy.append(entropy)
            per_layer_max.append(probs.max().item())

            mass_on_injection = probs[:prefix_len].sum().item()
            injection_mass.append(mass_on_injection)

        mean_injection_mass = sum(injection_mass) / len(injection_mass) if injection_mass else 0
        entropy_std = torch.tensor(per_layer_entropy).std().item() if per_layer_entropy else 0

        coherence = 1.0 - min(entropy_std / 2.0, 1.0)

        reading = GeometryReading(
            timestamp=time.time(),
            injection_label=label or cache_block.label,
            injection_tokens=prefix_len,
            query_tokens=input_ids.shape[1],
            per_layer_entropy=per_layer_entropy,
            per_layer_max_attention=per_layer_max,
            injection_attention_mass=injection_mass,
            coherence_score=coherence,
            metadata={
                'mean_injection_mass': mean_injection_mass,
                'entropy_std': entropy_std,
                'query_preview': query[:80],
            },
        )

        self.readings.append(reading)
        logger.info(
            f'Geometry: {label} → coherence={coherence:.3f}, '
            f'injection_mass={mean_injection_mass:.3f}, '
            f'layers_attended={sum(1 for m in injection_mass if m > 0.3)}/{len(injection_mass)}'
        )

        return reading

    def compare_with_without(self, query: str, cache_block,
                              label: str = '') -> dict:
        """Compare attention geometry with and without KV injection.

        Returns delta metrics showing what the injection changed.
        """
        with_reading = self.observe_injection(query, cache_block, f'{label}+injection')

        input_ids = self.tokenizer.encode(query, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs_bare = self.model(
                input_ids=input_ids,
                output_attentions=True,
                return_dict=True,
            )

        bare_entropy = []
        for layer_attn in outputs_bare.attentions:
            avg = layer_attn.mean(dim=1).squeeze(0)
            probs = avg[-1]
            entropy = -(probs * (probs + 1e-10).log()).sum().item()
            bare_entropy.append(entropy)

        entropy_deltas = [
            w - b for w, b in zip(with_reading.per_layer_entropy, bare_entropy)
        ]

        return {
            'with_injection': with_reading,
            'bare_entropy': bare_entropy,
            'entropy_delta': entropy_deltas,
            'mean_entropy_increase': sum(entropy_deltas) / len(entropy_deltas),
            'layers_most_affected': sorted(
                range(len(entropy_deltas)),
                key=lambda i: abs(entropy_deltas[i]),
                reverse=True
            )[:5],
        }

    def export_for_oracle(self) -> list[dict]:
        """Export readings in Oracle persistence format."""
        return [
            {
                'timestamp': r.timestamp,
                'label': r.injection_label,
                'injection_tokens': r.injection_tokens,
                'coherence': r.coherence_score,
                'per_layer_entropy': r.per_layer_entropy,
                'injection_attention_mass': r.injection_attention_mass,
                'metadata': r.metadata,
            }
            for r in self.readings
        ]
