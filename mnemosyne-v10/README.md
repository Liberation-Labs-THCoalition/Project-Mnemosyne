# mnemosyne-v10: whole-session retrieval

Mnemosyne v10 as a library and a command-line tool. It indexes a conversation history (sessions of turns)
and, for each question, returns the sessions most likely to hold the answer, whole, rendered in
chronological order, under a token budget. It makes no LLM calls.

The defaults are the frozen configuration of the published evaluation (config hash `7bc75f509aa0`), and a
parity test shows that this library reproduces that run's held-out retrieval record question by question.

## Results (from the paper; this directory adds no new benchmark numbers)

Paper, code and data: *Whole Sessions, Not Passages: Mnemosyne v10 Scores 96.2–96.8% on Held-Out
LongMemEval_S-cleaned Under Two Claude Judges*, in
[published-research/mnemosyne-longmemeval-v10](https://github.com/Liberation-Labs-THCoalition/published-research/tree/master/mnemosyne-longmemeval-v10).

- **Held-out accuracy:** **96.2% / 96.8%** (claude-sonnet-5 / claude-opus-5-5 judges) on the 400
  pre-registered held-out questions of LongMemEval_S-cleaned. The reader was claude-opus-5-5 with thinking.
- **All 500 questions:** **95.8% / 96.8%** (same judges), 100 of which were the retrieval-tuning split.
- **Evidence recall (no reader involved):** every `has_answer`-labelled evidence session is in the context
  on **97.1%** of held-out questions, and **96.0%** when every official `answer_session_ids` session is
  required.
- **Placement:** these are the highest point estimates among the LongMemEval_S results in our survey that
  we could trace to a primary source. On all 500, the claude-sonnet-5-judge figure is one question above the
  best published claims, and every one of our 95% intervals includes those claims.

The judges are Claude models using the official LongMemEval answer-check prompts, not the official GPT-4o
judge, and the dataset is the S-cleaned variant. The paper lists every deviation from the official harness.

## How it retrieves (frozen v10)

| step | what | frozen value |
|---|---|---|
| lexical channel | BM25 over **every** turn (user and assistant); Porter-stemmed ASCII terms, English stop words and 1-character terms dropped | k1 = 1.2, b = 0.75 |
| dense channel | cosine between the question and each **user** turn; BAAI/bge-base-en-v1.5 at revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, CLS pooling, L2-normalised, 512-token inputs, bge's retrieval prefix on the question only; vectors stored as float16 | |
| fusion | weighted reciprocal-rank fusion per turn: weight / (k + rank); ties broken by content hash, not position | k = 60, BM25 1.0, dense 2.0 |
| session score | best fused turn + second-best | `second_weight` 1.0 |
| packing | sessions in score order, each **whole** if it fits the remaining budget; otherwise a window of ±2 turns around its two best turns, if that fits | budget 24,000 o200k tokens |
| rendering | chosen sessions (or windows) in **chronological** order, with dates, in the history format of the official LongMemEval prompt builder | |

The time channel (`"time"`) from the dev sweep is available but off, as in the frozen config.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate   # recent Debian/Ubuntu refuse a system-wide pip install
pip install "./mnemosyne-v10[encoder]"     # library + CLI + the bge encoder (torch, transformers)
pip install "./mnemosyne-v10"              # without the encoder: BM25-only, or precomputed vectors
```

Dependencies: Python ≥ 3.10, `numpy` ≥ 2.0 (NumPy 1.x promotes float32 scalars differently, so some session
scores would not be computed as in the published run; parity was verified on 2.4.4), `tiktoken` (o200k_base;
downloads its encoding file on first use), `nltk` (the Porter stemmer only, no corpora). The encoder extra
adds `torch` and `transformers` and downloads bge-base-en-v1.5 (about 440 MB) at the pinned revision on
first use. scikit-learn is **not** needed: its English stop-word list is vendored
(`mnemosyne_v10/stopwords.py`) and checked against scikit-learn in the tests. (Importing nltk loads
scikit-learn anyway if it happens to be installed; nltk does not require it.) Without installing,
`PYTHONPATH=mnemosyne-v10 python -m mnemosyne_v10 ...` also works.

If loading the encoder fails with `operator torchvision::nms does not exist`, the installed torchvision was
built for a different torch. bge does not use torchvision: fix or remove it, or set
`MNEMOSYNE_V10_HIDE_TORCHVISION=1` to hide it from the process that loads the encoder.

## Library

```python
from mnemosyne_v10 import MemoryIndex, BgeEncoder

index = MemoryIndex(encoder=BgeEncoder())            # frozen v10 defaults; the encoder runs on CPU
index.add_session(
    [{"role": "user", "content": "My sister Bea adopted a greyhound called Pixel."},
     {"role": "assistant", "content": "Congratulations to Bea!"}],
    date="2026/09/20 (Sun) 18:30")                   # any string that sorts chronologically
index.save("./v10-index")                            # history.json + emb/ (user-turn vectors)

index = MemoryIndex.load("./v10-index", encoder=BgeEncoder())
r = index.retrieve("What is my sister's dog called?")
r.text          # the rendered history block to put in the reader's context
r.tokens        # budget tokens used (<= r.budget, 24,000 by default)
r.sessions      # SelectedSession(index, date, turns, turn_indexes, whole), chronological
```

- `retrieve(question, budget=8000)` changes the budget for one call; `MemoryIndex(config={...})` overrides
  any frozen value for the index (unknown keys raise).
- `MemoryIndex(config={"channels": ["bm25"]})` needs no encoder. It is not the evaluated configuration.
- Only `role` and `content` of each turn are read; other keys (ids, labels) are dropped at the door.
  Turns with role `user` are embedded; `assistant` turns reach the ranking through BM25 and their session.
- New sessions: `add_session()` embeds only user turns whose vectors are not already stored, and `save()`
  rewrites the index directory (atomically, file by file).

## CLI

```bash
python -m mnemosyne_v10 index    --history history.json --out ./v10-index     # {"sessions": [{"date", "turns"}]}
python -m mnemosyne_v10 add      --index ./v10-index --session session.json   # {"date", "turns"}
python -m mnemosyne_v10 retrieve --index ./v10-index --question "What is my sister's dog called?"
python -m mnemosyne_v10 retrieve --index ./v10-index --question - --format json --budget 12000 < q.txt
```

Every path is an argument. `--set KEY=VALUE` overrides a config value (JSON-parsed, e.g.
`--set 'channels=["bm25"]'`), `--config FILE` a set of them. `--device` picks the encoder's torch device
(default `cpu`). Installed with pip, the same commands are available as `mnemosyne-v10 ...`.

## Tests

```bash
cd mnemosyne-v10
python -m pytest tests/test_unit.py -q          # synthetic history, stub encoder; runs anywhere
MNEMOSYNE_V10_DATA=/path/to/longmemeval_s_cleaned.json \
MNEMOSYNE_V10_EMB=/path/to/v10/emb \
python -m pytest tests/test_parity.py -q        # skips cleanly without the two variables
```

**Unit tests** (`tests/test_unit.py`) need only numpy, nltk and pytest; the four checks that need tiktoken's
o200k file, scikit-learn or torch skip without them (`pip install "./mnemosyne-v10[encoder,test]"` runs all
35). They cover: frozen defaults and hash, and config validation; whole-session selection and chronological
rendering; the window fallback and `whole_top`; user-only embedding, cosine scores, and negative cosines
still ranked; the session score (best + second-best); tie-breaks by content hash and by v10's session hash;
term stemming and filtering, digits included; BM25's parameters as passed by `retrieve()`, and repeated
query terms counted once; `w_assistant`; v10's budget accounting; labels never reaching retrieval;
invariance to session order; save/load, including adding to a reopened index; BM25-only mode; the time
channel and its date windows; the exact rendering format; bge's query prefix and CLS pooling (with a
stand-in tokenizer and model); the CLI, including its JSON output; and BM25 and RRF checked bit-for-bit
against v10.py's own functions.

**Parity test** (`tests/test_parity.py`) runs the library over the 400 held-out questions with the published
run's precomputed embeddings and compares every question with `tests/fixtures/recall_heldout_frozen.json`
(the release's `data/recall_heldout_frozen.json`, sha256-checked): evidence coverage, budget tokens, session
count and whole-session count must all match. It also asserts the totals, 365/376 on `has_answer` and
361/376 on `answer_session_ids`. Two controls (`w_dense` 2 → 1, budget 24,000 → 8,500) must be detected as
differences, so the check is shown to be able to fail. That shows the comparison is not vacuous, not how
fine it is: a change that only reorders tied turns can leave all but a few questions unchanged, and the
byte-for-byte comparison with the reader's prompts (below) is the finer check. The controls run on the
record's first 50 questions only: evaluating an unfrozen configuration on held-out questions is the peek
v10's pre-registration rules out for results, so the controls assert only that differences exist, and their
totals are not results. Optional extras: `MNEMOSYNE_V10_PROMPTS=<dir of qNNN.txt>` compares the rendered
history with the reader's prompt files byte for byte, and `MNEMOSYNE_V10_ENCODER_CHECK=1` embeds a small
sample on CPU and compares it with the stored vectors. The embeddings are the ones the release's
`code/embed_turns.py` writes (`turn_index.json`, `turns.f16.npy`, `questions.f16.npy`, `meta.json`); they
are not redistributed, and computing them for the whole dataset takes hours on CPU.

The same comparison from the command line (exit status 1 on any difference):

```bash
python -m mnemosyne_v10 longmemeval --data longmemeval_s_cleaned.json --embeddings EMB_DIR \
    --record tests/fixtures/recall_heldout_frozen.json
```

## Relation to the harness as run

`v10.py` in the paper's release is the benchmark harness exactly as run, with absolute paths, and stays
untouched there. This package ports its `retrieve()` into an index that is built once and queried per
turn, and keeps its arithmetic: the same fusion and packing, float32 scores, float16-stored vectors, and the
same hash tie-breaks. The implementation differences are: BM25 runs off an inverted index with its
statistics precomputed, adding query-term contributions in sorted order (v10.py iterated a Python set, whose
order varies between runs); RRF is vectorised (checked bit for bit against v10.py's loop in the unit tests);
the dense matrix is built once per index instead of per query; and token counts are cached per text instead
of per turn key. None of them changes a parity field on the 400 held-out questions.

## Caveats (from the paper and its reviews)

- **Budget accounting.** The budget counts each turn as `json.dumps(turn, ensure_ascii=False)` in o200k
  tokens (+1 per turn, +16 per session). The rendered history escapes non-ASCII characters (`\u00e9`), as
  the official builder does. For English that puts it a few percent over the budget (at most about 5% on
  LongMemEval_S-cleaned); for other scripts it can be several times the budget (in our checks, about 4× for
  Chinese and 7× for Russian). `r.render(ensure_ascii=False)` stays within a few percent, but is not what the
  benchmark reader saw.
- **English only.** BM25 counts only ASCII letters and digits as word characters, and bge-base-en-v1.5 is
  an English model.
- **Duplicate turns.** Turns with identical content still tie-break by array order, so a history with
  duplicated turns can see small selection changes when reordered.
- **Scope of the evidence.** Evaluated on LongMemEval_S-cleaned only (about 115k tokens of history per
  question). Larger histories (the M variant) are untested.
- **Encoder cost.** bge on CPU is slow for large backfills; embed once and keep the index directory.
- **Load cost.** `MemoryIndex.load()` re-tokenizes every stored turn: about 6 CPU-seconds for a history of
  1.1 million tokens on our server, before torch and the encoder load (about 15 more). Keep one index open
  in a long-running process rather than reloading it per question. A Claude Code hook reloads it on every
  prompt and can add at most 10,000 characters; see `AGENT_SETUP.md`.

## Files

| path | what |
|---|---|
| `mnemosyne_v10/retrieval.py` | `MemoryIndex`, scoring, packing, rendering |
| `mnemosyne_v10/encoders.py` | `BgeEncoder` (pinned revision), `EmbeddingStore` (v10's on-disk vector layout) |
| `mnemosyne_v10/config.py`, `frozen_v10.json` | the frozen configuration (byte-identical to the release's) |
| `mnemosyne_v10/longmemeval.py` | label-free `view()`, evidence scoring, parity comparison |
| `mnemosyne_v10/cli.py` | `index`, `add`, `retrieve`, `longmemeval` |
| `tests/` | unit tests, parity test, the frozen held-out record |
