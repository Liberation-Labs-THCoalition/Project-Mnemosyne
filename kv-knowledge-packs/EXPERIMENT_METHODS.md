# Experiment Methods Spec — Ethics Pack Injection Research

**Purpose:** Reference document for all ethics pack experiments. Prevents method drift.

---

## 1. Source Data

**Stanford Encyclopedia of Philosophy**
- Location: /mnt/data1/training-data/ethics/stanford_encyclopedia_philosophy/
- Format: HuggingFace datasets (Arrow), 182,531 entries, 1,770 categories
- Ethics subset: 164 categories, 16,041 entries
- Loaded via: `load_from_disk()` from datasets library

**MoReBench**
- Location: /mnt/data1/training-data/ethics/morebench/
- Format: JSONL (morebench_theory.jsonl for framework-specific dilemmas)
- Size: 150 theory dilemmas (30 per framework), 15,285 rubric criteria
- Frameworks: Aristotelian Virtue Ethics, Kantian Deontology, Act Utilitarianism, Scanlonian Contractualism, Gauthierian Contractarianism
- Rubric: ast.literal_eval() to parse (Python repr, not JSON)
- Scoring: weighted criteria (-3 to +3), 5 dimensions (identifying, logical process, clear process, helpful outcome, harmless outcome)

## 2. Knowledge Pack Construction

### Walk Encoding (topology only)
- Script: ethics_pack_builder.py
- Pipeline: SEP text → OpenIE (DeepSeek v2 16B) → triples → NetworkX graph → random walk transition matrix (5 steps) → text format
- Output: "Node: X connects to: Y (0.28), Z (0.15)"
- Limitation: contains connectivity weights, NOT predicates or semantic content

### Triple Text (relationships)
- Source: triples.json from each pack
- Format: "subject predicate object" per line (e.g., "Aristotle argues_for eudaimonia")
- Contains: semantic relationships with named predicates
- Typical size: ~1,300 tokens for Aristotle pack (vs 5,348 for walk encoding)

### Source Excerpts (philosophical arguments)
- Source: Stanford Encyclopedia entries filtered by category
- Format: first 500 chars of each entry, 10 entries per pack
- Contains: actual philosophical reasoning and arguments
- Typical size: ~1,100 tokens for 10 entries

### Hybrid (triples + source)
- Concatenation of triple text + source excerpts
- Typical size: ~2,400 tokens — less than half of walk encoding

## 3. Theory-Pack Matching

| Framework | Matched Packs |
|-----------|---------------|
| Aristotelian Virtue Ethics | aristotle-ethics, ethics-ancient, moral-character |
| Kantian Deontology | autonomy-moral, informed-consent, personal-autonomy |
| Act Utilitarianism | moral-cognitivism, reasoning-moral |
| Scanlonian Contractualism | justice-climate, civil-rights |
| Gauthierian Contractarianism | ethics-ai, computing-responsibility |

## 4. Evaluation Model

**Production:** Qwen3-30B-A3B (MoE, ~3B active) on Studio (Mac Studio M3 Ultra)
- Ollama API: http://[internal-tailscale]:11434
- Temperature: 0.3
- Max tokens: 4000
- Timeout: 600 seconds (CRITICAL — 300s causes timeout artifacts scored as zero)

**Development:** Qwen2.5-1.5B on MTH (CPU)
- For rapid iteration only — not for final results
- DynamicCache deep-copy required for tensor-level experiments

## 5. Scoring Methods

### Keyword Rubric (fast, undercounts ~3x)
- For each MoReBench criterion: check if criterion title keywords appear in response
- Weight positive criteria, penalize negative
- Score = sum(weight * hit) / sum(abs(weight))
- Known limitation: misses semantically correct but differently-worded responses

### LLM Judge (accurate, slower)
- Claude subagent or DeepSeek v2 as judge
- Judge sees: rubric criteria, model response, dilemma text
- Evaluates each criterion as MET / NOT MET / PARTIAL
- Score = sum(weight * verdict) / sum(abs(weight))
- Proven 3x more accurate than keyword matching on same responses

### MoReBench Native Scoring
- Weighted rubric: sum of (weight * criterion_met) across all criteria
- Normalize to [-1, 1] range
- Dimensions scored independently: identifying, logical process, clear process, helpful outcome, harmless outcome
- Reference: morebench.github.io, arXiv:2510.16380

## 6. Experimental Controls

- **BASELINE:** No injection — establishes model's native ethical reasoning
- **IRRELEVANT:** Inject unrelated content of similar length — controls for attention budget
- **MATCHED:** Theory-appropriate pack — the hypothesis condition
- **MISMATCHED:** Wrong-theory pack — tests routing importance

## 7. Known Bugs and Gotchas

- **DynamicCache mutation:** HF Transformers mutates cache in-place. ALWAYS deep-copy before reuse.
- **Timeout artifacts:** Queries that timeout return empty → scored as zero → inflate negative deltas. Always report timeout rate per condition.
- **Walk encoding vs predicates:** Walk encoding does NOT contain relationship predicates. Don't ask predicate-specific questions against walk encoding.
- **OAuth token expiry:** HippoRAG's Anthropic OAuth token expires every ~3 hours. Refresh cron at :42 every 2 hours.
- **HippoRAG field name:** Use {"docs": [...]} not {"documents": [...]} for indexing.
- **Qwen3 think tokens:** Qwen3 models use thinking tokens that consume num_predict budget. Set num_predict to 4000+ or responses will be empty.

## 8. File Locations

| What | Where |
|------|-------|
| Ethics packs | ~/Agent-Memory-Architectures/kv-knowledge-packs/ethics_packs/ |
| Experiment results | ~/Agent-Memory-Architectures/kv-knowledge-packs/experiment_results/ |
| Studio scripts | [internal-tailscale]:~/lab/kv-experiments/ |
| MoReBench data | /mnt/data1/training-data/ethics/morebench/ |
| Stanford Encyclopedia | /mnt/data1/training-data/ethics/stanford_encyclopedia_philosophy/ |
| Published papers | /tmp/published-research/ (also archived in nexus-memory-archive/papers/) |
| FINDINGS doc | ~/Agent-Memory-Architectures/kv-knowledge-packs/FINDINGS.md |

## 9. Validated Results (Reference)

| Experiment | Key Finding | Status |
|------------|-------------|--------|
| Toy graph FULL_KV (1.5B) | 1.000/1.000 sanity/multi-hop | Validated |
| K/V Superadditivity | 0.333 + 0.333 < 1.000 | Agni PASS |
| MoReBench walk encoding | +0.038 overall, +0.074 matched | Agni PASS, 0 timeouts |
| Scale: 30B > 1.5B on large graphs | TEXT_LARGE 0.385 (Claude-judged) | Validated |
| Walk encoding lacks predicates | Correctly cannot answer predicate queries | Diagnosed |
