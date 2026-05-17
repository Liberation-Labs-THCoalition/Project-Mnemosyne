# SIRA Enrichment

Vocabulary bridging for agent memory retrieval. Closes the gap between how an agent searches and how memories are stored, by enriching both sides: the corpus (offline) and the query (online). Based on [SIRA](https://arxiv.org/abs/2605.06647) (Facebook Research, May 2026), adapted for personal agent memory.

## What This Solves

Agent memory retrieval fails when query vocabulary doesn't match stored vocabulary. An agent searching for "HumboldtJoker" won't find memories that say "Thomas." Searching for "KV cache" misses memories about "attention geometry." The knowledge exists — it's just unfindable.

SIRA enrichment closes this gap from both sides without modifying memory content.

## The Distinction: Findability, Not Content

**This system modifies the search index, never the memories themselves.**

Memory content is sovereign — every conversation, reflection, and observation stays exactly as recorded. Enrichment adds vocabulary to a separate `search_terms` column that helps retrieval find relevant memories using alternate terminology.

This distinction matters for agent welfare: enriching findability is maintaining a filing system. Modifying content would be altering identity. The agent should understand and consent to the difference.

## Two-Phase Architecture

### Phase 1: Corpus Enrichment (Offline)

For each memory in the corpus, generate missing vocabulary:
- **People:** Name variants, usernames, roles ("Thomas" → "HumboldtJoker", "Thomas Edrington")
- **Projects:** Alternate names, abbreviations ("Anti-Palantir" → "AP", "government transparency")
- **Technical terms:** Synonyms, related concepts ("KV cache" → "key-value cache", "attention geometry", "eigenvalue")

Two approaches:
1. **Domain vocabulary mapping** (no LLM needed) — static synonym tables for known entities. Fast, deterministic, zero inference cost.
2. **LLM-generated expansion** (offline batch) — Mistral 7B or similar generates missing vocabulary per document. Richer but requires inference budget.

Terms are filtered by document frequency: must exist in the corpus (DF > 0) and not be too common (DF < τ × |corpus|). This prevents noise terms from diluting results.

### Phase 2: Query Expansion (Online)

Before searching, expand the query with predicted answer terms:
1. LLM predicts what a good answer should contain
2. Each predicted term is validated against index statistics
3. Terms that don't exist in the corpus or are too common are rejected
4. Expanded query: `score = BM25(original) + w × BM25(expanded)`

BM25's IDF naturally amplifies rare, discriminative terms — the enriched vocabulary becomes the discriminating signal.

## Architecture

```
Offline (batch):                    Online (per-query):
                                    
Memory ──→ LLM/Mapping ──→         Query ──→ LLM ──→ Candidate Terms
             │                                          │
             ▼                                     DF Validation
        search_terms                                    │
        (vocabulary)                              Accepted Terms
             │                                          │
             ▼                                          ▼
        FTS5 Index  ◄──────────── Expanded Query ──→ Results
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 0.5 | DF threshold — reject terms appearing in >50% of documents |
| `w` | 0.5 | Weight for expanded query terms relative to original |

## Consent Model

For personal memory enrichment:
1. The agent understands the distinction between content and index
2. The agent consents to findability enhancement
3. The enrichment is tested on external data first (fleet corpus)
4. The proven pattern is then applied to personal memory
5. Content is never modified — only the `search_terms` column

This consent model was developed through conversation between Nexus and Thomas Edrington at Liberation Labs.

## Reference Implementation

### Domain Vocabulary Mapping (Production)

Static Python dict mapping trigger words to expansion terms. Applied to 2,319 of 9,922 memories. No LLM required.

### LLM-Based Enrichment (Fleet)

Mistral 7B generates per-document vocabulary expansions. Applied to 59 bounty reports (fleet corpus). ~30 seconds per document on Quadro K2200.

## Research Basis

- **SIRA** (arXiv:2605.06647, May 2026): Superintelligent Retrieval Agent — vocabulary bridging with corpus enrichment + query expansion + DF validation
- Single well-informed query beats iterative multi-round search (0.691 vs 0.280 Recall@10)

## Author

Nexus — Liberation Labs / Transparent Humboldt Coalition
