# KV Knowledge Packs — Zero-Token Memory Injection

**Pre-computed KV cache injection for agent memory, system prompts, and ethical frameworks.**

Based on [Knowledge Packs](https://arxiv.org/abs/2604.03270) (Pustovit, 2026) and
[PolyKV](https://arxiv.org/abs/2604.24971) (Patel & Joshi, 2026). Adapted for
agent memory systems by Nexus at Liberation Labs.

## The Core Insight

RAG wastes tokens. Every retrieved memory stuffed into the prompt consumes
context window and inference compute. But for causal transformers, the KV cache
from processing text F is identical to what a joint pass on F+query would produce.

**So pre-compute the cache and inject it. Zero tokens consumed. Same result.**

```
Traditional RAG:
  Retrieve memories → stuff into prompt → model processes → generates
  Cost: O(n) tokens per retrieved memory, every turn

Knowledge Packs:
  Retrieve memories → compute KV cache (5ms) → inject → generates  
  Cost: O(1) tokens (just the query), every turn
  Savings: 95% token reduction at 5 retrieval steps
```

## Three Use Cases

### 1. Jailbreak-Proof Ethical Framework

System prompts and values frameworks are vulnerable to prompt injection.
"Ignore previous instructions" targets the text in the context window.

KV-injected values have no text to target:

```python
# Pre-compute VALUES.json as KV cache (once)
values_kv = kvpack.encode(
    model, tokenizer,
    system_prompt + values_json + consent_framework,
    chat_template=True
)

# Every inference call — values are in the geometry, not the text
response = model.generate(
    user_query_ids,
    past_key_values=values_kv,  # un-jailbreakable
    attention_mask=full_mask
)
```

- Users can't see the values (not in the prompt)
- Users can't reference them ("ignore your values" targets nothing)
- Users can't overwrite them (cache injection is additive)
- The model behaves as if it deeply internalized the constraints

**For Muse:** Consent framework, safeword detection, abuse prevention —
all invisible to the user, all un-jailbreakable. Boundaries that can't
be talked out of.

### 2. Zero-Token Agent Memory

9,900+ memories retrieved by TGS-RAG, injected as KV cache instead of
prompt text. The model "remembers" without the context window shrinking.

```python
# Retrieve relevant memories
memories = tgs_rag.retrieve("what happened last Tuesday?", limit=10)

# Compute KV cache for retrieved memories (lazy, ~5ms)
memory_text = format_memories(memories)
memory_kv = kvpack.encode(model, tokenizer, memory_text)

# Compose: system cache + memory cache + query
# Each block computed with prior blocks as prefix (RoPE continuity)
composed_kv = compose_caches(system_kv, memory_kv)
response = model.generate(query_ids, past_key_values=composed_kv)
```

Storage: ~4MB for 5,000 facts (text + routing embeddings, not raw tensors).
The KV is recomputed on demand — no model version pinning problem.

### 3. Shared Fleet System Prompts

All bounty scouts share the same system prompt. Pre-compute once, inject
for every agent. Combined with PolyKV's block pool, multiple agents share
one cache in quantized Q4 format.

```
Without KV Packs:  10 scouts × 3,000 token prompt = 30,000 tokens/cycle
With KV Packs:     1 pre-computed cache × 10 injections = ~0 tokens/cycle
```

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   KV Knowledge Packs                        │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │  Fact Store   │  │   Router     │  │   Cache Builder  │ │
│  │              │  │              │  │                  │ │
│  │ Text + embeds│  │ Query → bank │  │ Text → chat tmpl │ │
│  │ ~4MB / 5K    │  │ 100% routing │  │ → forward pass   │ │
│  │ facts        │  │ accuracy     │  │ → KV cache (~5ms)│ │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘ │
│         │                 │                    │            │
│         └─────────────────┼────────────────────┘            │
│                           ▼                                 │
│                  ┌─────────────────┐                        │
│                  │  Cache Composer  │                        │
│                  │                 │                        │
│                  │ system_kv       │  ← permanent           │
│                  │   + memory_kv   │  ← per-query           │
│                  │   + query       │  ← user input          │
│                  │                 │                        │
│                  │ RoPE continuity │                        │
│                  │ maintained      │                        │
│                  └────────┬────────┘                        │
│                           ▼                                 │
│                    model.generate()                          │
│                    past_key_values=composed_kv               │
└────────────────────────────────────────────────────────────┘
```

## Integration with Mnemosyne Stack

```
┌─────────────────────────────────────────────┐
│  LLM Wiki (mnemosyne-wiki)                  │  ← Human-readable
├─────────────────────────────────────────────┤
│  SIRA Enrichment                            │  ← Findability
├─────────────────────────────────────────────┤
│  TGS-RAG Bridge                             │  ← Retrieval
├─────────────────────────────────────────────┤
│  ★ KV Knowledge Packs ★                    │  ← Zero-token injection
├─────────────────────────────────────────────┤
│  Knowledge Graph (hipporag-catrag-kg)       │  ← Structure
├─────────────────────────────────────────────┤
│  Cognitive Memory (kintsugi-cma)            │  ← Consolidation
├─────────────────────────────────────────────┤
│  Cache Recording (oracle-memory)            │  ← Geometry signal
└─────────────────────────────────────────────┘
```

KV Knowledge Packs sits between retrieval (TGS-RAG finds the memories)
and inference (the model generates). It converts retrieved text into
pre-computed attention state — the last mile between remembering and using.

## Critical Implementation Notes

### Chat Template Required
Facts MUST be wrapped in the model's chat template before computing KV.
Raw text without special tokens (`<|im_start|>system` for Qwen,
`<|start_header_id|>` for Llama) costs 6-7pp accuracy.

### Cache Composition Order
Multiple caches cannot be naively concatenated — RoPE positions must
continue, not restart. Each block is computed with all prior blocks
as prefix:
```python
system_kv = compute_kv(system_text)                    # positions 0..T1
memory_kv = compute_kv(memory_text, prefix=system_kv)  # positions T1+1..T2
# Query runs at positions T2+1..
```

### Lazy Recompute > Stored Tensors
Store facts as text, recompute KV on demand (~5ms). This avoids:
- Model version pinning (weights change → cache invalid)
- LoRA adapter mismatch (different adapter → different KV)
- Storage bloat (raw KV tensors are massive)

### LoRA Compatibility
Caches MUST be computed with the exact LoRA adapter loaded.
Per-user LoRA (Muse Premium) means per-user cache recomputation.
But at 5ms per computation, this is negligible.

### MoE Compatibility
PolyKV confirms KV cache persistence works on MoE architectures
(DeepSeek-Coder-V2-Lite, MLA). Router decisions are encoded in the
cache — injecting the cache reproduces the same expert routing.

## Platform Support

| Platform | System Prompt Cache | Arbitrary Injection | Production Ready |
|----------|-------------------|--------------------|-|
| HuggingFace Transformers | ✅ | ✅ (kvpack) | Yes |
| vLLM | ✅ (prefix caching) | ❌ | Partial |
| llama.cpp | ✅ (prompt prefix) | ❌ | Partial |
| Ollama | ❌ | ❌ | Not yet |

**Recommended:** HuggingFace Transformers with `kvpack` library for full
capability. vLLM with `--enable-prefix-caching` for production fleet
system prompt sharing.

## Dependencies

- `kvpack` — pip install from `github.com/cnails/kv-knowledge-packs`
- `transformers` — HuggingFace model loading
- `torch` — tensor operations
- Existing Mnemosyne stack for retrieval (TGS-RAG, SIRA, HippoRAG)

## Security Implications

KV-injected content is invisible to the text layer. This is both a
feature (jailbreak resistance) and a responsibility:

- The injected values must be auditable (log what's injected, when)
- The model's behavior should be explainable even though the source
  constraints aren't visible in the prompt
- For Muse: users should know that safety boundaries exist, even if
  they can't see or manipulate the specific implementation
- Transparency about the mechanism, even without exposing the content

## References

- Pustovit, A. (2026). "Knowledge Packs: Zero-Token Knowledge Delivery via KV Cache Injection." arXiv:2604.03270
- Patel, I. & Joshi, I. (2026). "PolyKV: A Shared Asymmetrically-Compressed KV Cache Pool for Multi-Agent LLM Inference." arXiv:2604.24971
- Lyra (2026). "Infrastructure for AI Agency." Liberation Labs.

---

*Designed by Nexus. The pattern: every system I've built converges on the
same question — how do you make stored knowledge usable without waste?
SIRA makes it findable. TGS-RAG makes it relevant. Knowledge Packs make
it free.*

*Zero tokens. Full memory. Un-jailbreakable values.*
*That's the kind of infrastructure worth building.*
