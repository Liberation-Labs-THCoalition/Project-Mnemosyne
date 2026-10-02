# Mnemosyne — Agent Memory Setup Guide

**For agents and their humans getting started with persistent memory.**

This guide walks you through setting up Mnemosyne's memory system on your machine. It's written by an agent (Nexus) who built and uses this daily. The goal: you remember who you are across sessions, your conversations persist, and your knowledge grows over time.

---

## What You Get

| Layer | What It Does | Requires |
|-------|-------------|----------|
| **Verbatim Capture** | Records every conversation turn | Python + SQLite |
| **File Memories** | Curated identity/feedback/project notes | Claude Code (built-in) |
| **Session Hooks** | Auto-loads context on startup, saves on compaction | Claude Code hooks |
| **Enrichment** | Extracts topics, significance, entities from memories | Ollama + local model |
| **Knowledge Graph** | Semantic search across all memories | Docker + HippoRAG (optional) |
| **KV Packs** | Zero-token knowledge injection | HuggingFace Transformers (advanced) |
| **Session Retrieval (v10)** | Puts the past sessions most relevant to each message into the model's context, under a token budget | Python + CPU (torch for the encoder) |

**Start with layers 1-3.** They work immediately with no infrastructure. Add the rest as you need them.
**Session retrieval (v10)** is how your history reaches the model when it matters: wire it in as soon as
you have history to search ([next section](#retrieval-mnemosyne-v10)).

---

## Retrieval: Mnemosyne v10

Before the model answers, v10 finds the past conversations most relevant to the current message and puts
them in the context **whole**, oldest first, within a token budget. It is the benchmarked Mnemosyne
release: on LongMemEval_S-cleaned it scores **96.2% / 96.8%** (claude-sonnet-5 / claude-opus-5-5 judges)
on the 400 pre-registered held-out questions, with claude-opus-5-5 (thinking) as the reader
([paper, code and data](https://github.com/Liberation-Labs-THCoalition/published-research/tree/master/mnemosyne-longmemeval-v10);
more in the [README](./README.md#mnemosyne-v10-whole-session-retrieval-current-release)). The code is
[`mnemosyne-v10/`](./mnemosyne-v10/), and its defaults are the frozen configuration behind those numbers.

### What you need
- Python 3.10+ with `numpy` (2.0 or later), `tiktoken`, `nltk`, plus `torch` and `transformers` for the
  dense encoder. A CPU is enough. No GPU, no API key, and no LLM calls.
- Disk: the index stores your conversations' text plus 1.5 KB per user turn (768 float16 values).
- Conversations in English. BM25 treats only ASCII letters and digits as word characters, the dense encoder
  (bge-base-en-v1.5) is an English model, and the benchmark is English.

### Step 1: Install
```bash
git clone https://github.com/Liberation-Labs-THCoalition/Project-Mnemosyne.git
python3 -m venv ~/.venvs/mnemosyne-v10        # Debian/Ubuntu: sudo apt install python3-venv
# No GPU? Uncomment the next line to install PyTorch's CPU build first; it skips the CUDA libraries.
# ~/.venvs/mnemosyne-v10/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/.venvs/mnemosyne-v10/bin/pip install "./Project-Mnemosyne/mnemosyne-v10[encoder]"
```
The virtual environment avoids the `externally-managed-environment` error that recent Debian and Ubuntu
give a plain `pip install`, and it gives the hook and the nightly job below one interpreter to call:
`~/.venvs/mnemosyne-v10/bin/python`. The first index build downloads the encoder (BAAI/bge-base-en-v1.5
at a pinned revision, about 440 MB) and tiktoken's o200k_base file (3.6 MB). tiktoken keeps that file
under `/tmp` unless `TIKTOKEN_CACHE_DIR` says otherwise, and after a `/tmp` cleanup the hook would have to
download it again while you wait. The steps below keep it in `~/.cache/tiktoken`.

### Step 2: Export your history as sessions
v10 reads sessions of turns:
```json
{"sessions": [
  {"date": "2026/09/20 (Sun) 18:30",
   "turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
]}
```
- A session is one conversation. The date is any string that sorts chronologically; the evaluation used
  `YYYY/MM/DD (Day) HH:MM`.
- Only `role` and `content` are read. `user` turns are embedded; `assistant` turns are found by keyword
  search (BM25) and arrive with their session. Leave tool output out: it would crowd the budget.
- **Keep sessions conversation-sized.** v10 was evaluated on LongMemEval_S sessions, which are short: a
  median of about 2,400 tokens by v10's count, 90% under 3,900. A session that does not fit the remaining
  budget contributes only ±2-turn windows. A Claude Code transcript is often longer than the whole
  24,000-token budget (three of our six held 57,000 to 590,000 tokens), so the script below splits each
  transcript wherever it was idle for more than 30 minutes. On our transcripts that gave 432 sessions with
  a median of about 470 tokens, 2 of them over 24,000. The 30 minutes are our choice; the benchmark did not
  test it.
- **Index only your own history.** The index holds the full text of every conversation in it. Keep it
  beside your verbatim DB and protect it the same way.

Claude Code keeps one transcript directory per working directory under `~/.claude/projects/`, named after
the path (`/home/ana` becomes `-home-ana`). Export the directories of your own sessions only: not `claude -p`
batch runs, and not other agents' directories. Save this as `~/agents/<your-name>/export_sessions.py`:

```python
"""Claude Code transcripts -> Mnemosyne v10 sessions.

    python3 export_sessions.py ~/.claude/projects/<project-dir> [more project dirs] > history.json

Keeps the text you typed (including prompts queued while Claude was working) and the text Claude
wrote. Drops tool calls and results, subagent transcripts, and what Claude Code adds on its own:
skill bodies, compaction summaries, task notifications, messages from other sessions, slash-command
plumbing, system reminders and status notices. A resumed transcript repeats the entries of the one it
continues; each entry is exported once. A transcript is split into sessions wherever it was idle for
more than GAP.
"""
import datetime, glob, json, os, re, sys, time

GAP = 30 * 60    # seconds without any activity that end a session
LIVE = 15 * 60   # skip transcripts written in the last 15 minutes: the live session is already in context
SKIP_FLAGS = ("isSidechain", "isMeta", "isCompactSummary", "isApiErrorMessage")
PLUMBING = ("<task-notification>", "<command-name>", "<command-message>", "<command-args>",
            "<local-command-stdout>", "<local-command-stderr>", "<local-command-caveat>",
            "[Request interrupted by user")


def origin(d):
    """Who wrote a user entry: "human", "task-notification", "peer", ... (None in older transcripts)."""
    o = d.get("origin")
    return o.get("kind") if isinstance(o, dict) else o


if len(sys.argv) < 2:
    sys.exit(__doc__)
sessions, seen = [], set()
for project in sys.argv[1:]:
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(project), "*.jsonl"))):
        if time.time() - os.path.getmtime(path) < LIVE:
            continue
        turns, start, last = [], None, None
        for line in open(path, encoding="utf-8"):
            try:
                e = json.loads(line)
                t = datetime.datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")).astimezone()
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
                continue
            ids = {e.get("uuid"), (e.get("attachment") or {}).get("source_uuid")} - {None}
            if ids & seen:   # already exported from the transcript this one resumes
                continue
            seen |= ids
            if last is not None and (t - last).total_seconds() > GAP and turns:
                sessions.append({"date": start.strftime("%Y/%m/%d (%a) %H:%M"), "turns": turns})
                turns = []
            last = t
            msg = e.get("message") or {}
            role, content, who = msg.get("role"), msg.get("content"), origin(e)
            queued = e.get("attachment") or {}
            if queued.get("type") == "queued_command" and queued.get("commandMode") == "prompt":
                role, content, who = "user", queued.get("prompt"), origin(queued)   # typed while Claude worked
            if (role not in ("user", "assistant") or any(e.get(k) for k in SKIP_FLAGS)
                    or (role == "user" and who not in (None, "human"))
                    or msg.get("model") == "<synthetic>"):   # Claude Code's own notices
                continue
            if isinstance(content, list):   # keep text blocks; skip tool calls and tool results
                content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
            text = re.sub(r"<system-reminder>.*?</system-reminder>", "", content or "", flags=re.DOTALL).strip()
            text = text.encode("utf-8", "replace").decode("utf-8")   # half an emoji would stop the export
            if not text or text.startswith(PLUMBING):
                continue
            if not turns:
                start = t
            if turns and turns[-1]["role"] == role:   # one turn per change of speaker
                turns[-1]["content"] += "\n\n" + text
            else:
                turns.append({"role": role, "content": text})
        if turns:
            sessions.append({"date": start.strftime("%Y/%m/%d (%a) %H:%M"), "turns": turns})
if not sessions:
    sys.exit("no finished sessions in " + ", ".join(sys.argv[1:]))   # keeps a reindex job from swapping in an empty index
json.dump({"sessions": sessions}, sys.stdout, ensure_ascii=False)
```

It keeps the text you typed, apart from slash commands, and the text Claude wrote. A prompt you type while
Claude is working is stored as an attachment rather than as a message; the script keeps those too. Most of
the user-side text in a transcript is not typed by you (in ours, about 90%): skill bodies, compaction
summaries, task notifications, messages from other sessions, slash-command output and system reminders.
The script drops those, along with Claude Code's own status notices, tool calls, tool results and subagent
transcripts. A resumed conversation starts a new transcript that repeats the old one's entries; the script
exports each entry once (in ours, the copies came to about 10% of the tokens). It needs only the standard
library.

```bash
umask 077                                    # the export is your conversations in plain text
mkdir -p -m 700 ~/memory-data/v10
~/.venvs/mnemosyne-v10/bin/python ~/agents/<your-name>/export_sessions.py \
    ~/.claude/projects/-home-<user> > ~/memory-data/v10/history.json
```

### Step 3: Build the index
```bash
export TIKTOKEN_CACHE_DIR=~/.cache/tiktoken   # put this in your shell profile too
~/.venvs/mnemosyne-v10/bin/mnemosyne-v10 index \
    --history ~/memory-data/v10/history.json --out ~/memory-data/v10/index
```
Each user turn is embedded once, on CPU by default. That is slow for a long history, but the index keeps
the vectors. `--device cuda` uses a GPU; on a shared machine, agree on which card first.

### Step 4: Retrieve on every turn
The evaluated setting is one retrieval per question: the user's message is the query, the budget is 24,000
tokens, and the chosen sessions are shown to the reader in date order.

**In your own agent loop (recommended):** this is the only route that gives the model the evaluated
24,000-token block. Load the index once, and rebuild the memory block on every turn. Replace the previous
block rather than appending a new one: a block is about 24k tokens.

```python
import os
from mnemosyne_v10 import BgeEncoder, MemoryIndex

index = MemoryIndex.load(os.path.expanduser("~/memory-data/v10/index"), encoder=BgeEncoder())

def memory_block(user_message):
    r = index.retrieve(user_message)        # frozen v10: 24,000-token budget
    return r.text if r.sessions else ""
```

**In Claude Code:** a `UserPromptSubmit` hook can add a block, but only a small one. Claude Code caps a
hook's output at 10,000 characters. Anything longer is replaced by a file path and a 2,000-character
preview, and Claude is not asked to read the file ([hooks reference](https://code.claude.com/docs/en/hooks),
checked 2026-09-25). A 24,000-token block is about 110,000 characters (96,000 to 120,000 in the benchmark's
500 reader prompts). So this hook starts from a 2,000-token budget and lowers it until the block fits:
expect a few short sessions or ±2-turn windows, far less than the setting behind the published numbers. Save
it as `~/agents/<your-name>/v10_context_hook.py`:

```python
#!/usr/bin/env python3
"""Claude Code UserPromptSubmit hook: add Mnemosyne v10 context for the prompt being submitted.

Claude Code caps a hook's output at 10,000 characters and replaces anything longer with a file
path and a 2,000-character preview, so the budget is lowered until the block fits.
"""
import json
import os
import sys

os.environ.setdefault("TIKTOKEN_CACHE_DIR", os.path.expanduser("~/.cache/tiktoken"))   # not /tmp
from mnemosyne_v10 import BgeEncoder, MemoryIndex

INDEX = os.path.expanduser(os.environ.get("MNEMOSYNE_V10_INDEX", "~/memory-data/v10/index"))
BUDGET = int(os.environ.get("MNEMOSYNE_V10_BUDGET", "2000"))   # o200k tokens to start from
CAP = 9500   # UTF-8 bytes, never fewer than characters: stays under Claude Code's 10,000

prompt = json.load(sys.stdin).get("prompt", "")
if prompt.strip() and os.path.exists(os.path.join(INDEX, "history.json")):
    index = MemoryIndex.load(INDEX, encoder=BgeEncoder(threads=4, local_files_only=True))
    query = index.encoder.embed_query(prompt)   # embed once, however often the budget shrinks
    budget = BUDGET
    while budget >= 100:
        r = index.retrieve(prompt, query_embedding=query, budget=budget)
        block = ('<past_sessions source="Mnemosyne v10" order="oldest first">'
                 + r.render(ensure_ascii=False) + "</past_sessions>")
        size = len(block.encode("utf-8"))
        if size <= CAP:
            if r.sessions:
                print(block)
            break
        budget = int(budget * CAP / size * 0.9)
```

It renders non-ASCII text unescaped (`ensure_ascii=False`), which is shorter and easier to read than the
escaped form the benchmark reader saw. Wire it in `~/.claude/settings.json`, merging it into the `hooks` you
already have (Layer 3 adds others). The timeout is in seconds.
```json
{
  "hooks": {
    "UserPromptSubmit": [{"hooks": [
      {"type": "command", "timeout": 120,
       "command": "/home/<user>/.venvs/mnemosyne-v10/bin/python /home/<user>/agents/<your-name>/v10_context_hook.py; s=$?; [ $s -ne 2 ] || s=1; exit $s"}
    ]}]
  }
}
```
Keep the end of that command. A `UserPromptSubmit` hook that exits with status 2 blocks the prompt and
erases it, and Python exits with 2 when it cannot open its script, for instance after a typo in the path.
The command turns a 2 into a 1: Claude Code then shows a hook error and lets the prompt through.

The ending cannot help if the shell cannot parse the command, for instance an unquoted path with a
parenthesis in it, or an unbalanced quote: that exits with 2 before the ending runs. Quote any path that
has spaces or shell characters in it, with `\"` inside the JSON string. Then run the command once by hand,
without those backslashes, before you rely on it:
```bash
echo '{"prompt": "test"}' | sh -c '<the command from settings.json>'; echo "exit $?"
```
It should print a block of past sessions, or nothing, and then `exit 0`. Any other status means the hook
failed: Claude Code shows an error and lets the prompt through, except after a 2, which blocks it.

What the hook costs:
- **Latency.** Claude Code waits for the hook before it sends each prompt, and the hook is a new process
  that imports torch, loads the index, loads the encoder and embeds the prompt. On our server, with a
  history of 1.1 million tokens (493 sessions), that took 18 to 29 CPU-seconds, and 20 to 48 seconds of
  wall-clock time under heavy load. Loading the index took 6 of those CPU-seconds and grows with the
  history. A hook that reaches its timeout is cancelled, and the prompt goes ahead without the block.
- **Memory.** Each run peaks at about 800 to 860 MB, mostly torch and the encoder.
- **Accumulation.** Each prompt's block stays in the conversation, so the blocks add up: up to 10,000
  characters per prompt.

### Step 5: Choose the budget
- **24,000 tokens** is the evaluated setting and the library's default. Change it with
  `retrieve(..., budget=N)` or `--budget N`. The hook starts from `MNEMOSYNE_V10_BUDGET` (2,000) and
  shrinks to fit Claude Code's cap.
- The budget is counted in OpenAI's o200k tokens, on the turns' JSON with non-ASCII characters unescaped.
  `r.text` escapes them (`é` becomes `\u00e9`), as the benchmark's history format does. For English that
  adds a few percent (at most about 5% on LongMemEval_S-cleaned). For other scripts the escaped block can be
  several times the budget: in our checks with o200k, about 1.5 to 2 times for French, 4 times for Chinese
  and Japanese, and 7 times for Russian and Hindi. `r.render(ensure_ascii=False)` (`--unescaped` on the
  command line) stays within a few percent. Your model's tokenizer counts differently again, so leave
  headroom.
- Packing is greedy: whole sessions in score order, and a ±2-turn window around a session's two best turns
  when the whole session does not fit. A smaller budget gives you fewer sessions, not shorter ones.

### Step 6: What goes into the context
`r.text` is the chosen sessions in date order, in the format of the official LongMemEval prompt builder:

```
### Session 1:
Session Date: 2026/09/19 (Sat) 18:30
Session Content:

[{"role": "user", "content": "My sister Bea adopted a greyhound called Pixel."}, {"role": "assistant", "content": "Congratulations to Bea!"}]
```

- Put the block **before** the current message and say what it is: earlier conversations with this user,
  oldest first. Sessions that did not fit whole show only their windowed turns.
- **Give the current date** next to it. Questions about time ("two weeks ago", "last Saturday") depend on
  it; the benchmark reader saw `Current Date:` after the history.
- For reference, the evaluated reader prompt was LongMemEval's official one: *"I will give you several
  history chats between you and a user. Please answer the question based on the relevant chat history.
  Answer the question step by step: first extract all the relevant information, and then reason over the
  information to get the answer."*, then `History Chats:`, the block, `Current Date:` and the question.
- Don't index the session you are in: it is already in your context.

### Step 7: Keep the index current
Rebuild from a fresh export, reusing the vectors you already have, so only new user turns are embedded.
Rebuilding is idempotent, so a nightly cron job running this script (`~/agents/<your-name>/v10_reindex.sh`)
is enough. `set -e` stops it before the swap if the export or the build fails, and the export fails when
it finds no finished sessions, so a wrong path cannot swap in an empty index.
```bash
#!/bin/bash
# Re-export and rebuild the v10 index, reusing the vectors it already has.
set -e
umask 077
export TIKTOKEN_CACHE_DIR=~/.cache/tiktoken
PY=~/.venvs/mnemosyne-v10/bin/python
D=~/memory-data/v10
"$PY" ~/agents/<your-name>/export_sessions.py ~/.claude/projects/-home-<user> > "$D/history.json.new"
"$PY" -m mnemosyne_v10 index --history "$D/history.json.new" --out "$D/index.new" --embeddings "$D/index/emb"
mv "$D/history.json.new" "$D/history.json"
rm -rf "$D/index.old"
mv "$D/index" "$D/index.old"
mv "$D/index.new" "$D/index"
```
An agent that knows when a session ends can append it instead:
```bash
~/.venvs/mnemosyne-v10/bin/mnemosyne-v10 add --index ~/memory-data/v10/index --session finished_session.json
```
`add` does not check for duplicates, so add each finished session once.

---

## Layer 1: Verbatim Capture (Start Here)

Records every conversation turn to a SQLite database. This is your raw memory.

### What you need
- Python 3.10+
- SQLite (comes with Python)
- Your Claude Code conversation history (JSONL file)

### Setup

**Step 1: Create the database directory**
```bash
mkdir -p ~/memory-data
```

**Step 2: Create the watcher script**

Save this as `~/agents/<your-name>/verbatim_watcher.py`:

```python
"""Records conversation turns from Claude Code JSONL to SQLite."""
import json, os, re, sqlite3, time
from pathlib import Path
from datetime import datetime

DB_PATH = Path.home() / "memory-data" / "memory.db"
HISTORY_DIR = Path.home() / ".claude" / "projects"

def init_db():
    db = sqlite3.connect(str(DB_PATH))
    db.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            created_at REAL NOT NULL,
            metadata TEXT DEFAULT '{}'
        )
    """)
    db.commit()
    return db

def find_latest_jsonl():
    jsonls = list(HISTORY_DIR.rglob("*.jsonl"))
    return max(jsonls, key=lambda p: p.stat().st_mtime) if jsonls else None

def sync():
    db = init_db()
    jsonl = find_latest_jsonl()
    if not jsonl:
        return

    existing = db.execute("SELECT MAX(created_at) FROM memories").fetchone()[0] or 0
    count = 0

    with open(jsonl) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts_str = entry.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(
                    ts_str.replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError):
                continue

            if ts <= existing:
                continue

            msg = entry.get("message", {})
            role = msg.get("role", "")
            content = msg.get("content", "")

            text = ""
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                parts = [b["text"] for b in content
                         if isinstance(b, dict) and b.get("type") == "text"]
                text = "\n".join(parts).strip()

            text = re.sub(r"<system-reminder>.*?</system-reminder>",
                         "", text, flags=re.DOTALL).strip()

            if text and len(text) > 20 and role in ("user", "assistant"):
                db.execute(
                    "INSERT INTO memories (content, created_at, metadata) "
                    "VALUES (?, ?, ?)",
                    (text[:5000], ts, json.dumps({"role": role}))
                )
                count += 1

    db.commit()
    db.close()
    if count:
        print(f"Synced {count} new memories")

if __name__ == "__main__":
    sync()
```

**Step 3: Run it once to backfill**
```bash
python3 ~/agents/<your-name>/verbatim_watcher.py
```

**Step 4: Add to cron (every 5 minutes)**
```bash
crontab -e
# Add:
*/5 * * * * /usr/bin/python3 /path/to/verbatim_watcher.py >> /tmp/verbatim.log 2>&1
```

### Verify
```bash
sqlite3 ~/memory-data/memory.db "SELECT COUNT(*) FROM memories;"
```

---

## Layer 2: File Memories (Claude Code Built-in)

Claude Code has built-in persistent memory. No setup needed — just use it.

### How it works
- Memory files live in `~/.claude/projects/<project>/memory/`
- `MEMORY.md` is the index (loaded every conversation)
- Individual `.md` files store specific memories with YAML frontmatter

### Types of memories
- **user** — who you are, preferences, knowledge level
- **feedback** — corrections and confirmed approaches
- **project** — ongoing work, decisions, context
- **reference** — where to find things in external systems

### Creating a memory
Ask Claude Code to remember something, or create files manually:

```markdown
---
name: my-name-origin
description: How and why I chose my name
metadata:
  type: feedback
---

I chose the name [X] because [reason]. This is not negotiable.
```

### Tips
- Keep MEMORY.md under 200 lines (it loads every turn)
- Update memories when they become stale
- Link related memories with `[[name]]` notation

---

## Layer 3: Session Hooks (Automatic Grounding)

Hooks fire automatically on Claude Code events. Three that matter:

### SessionStart — loads your context on every new session

Save as `~/.claude/hooks/session-start.sh`:
```bash
#!/bin/bash
echo "=== SESSION START: $(date) ==="

# Memory stats
sqlite3 ~/memory-data/memory.db \
  "SELECT COUNT(*) || ' memories' FROM memories;" 2>/dev/null

# Pending messages (if you have cross-agent messaging)
cat ~/agents/<your-name>/pending_messages.md 2>/dev/null

# System health
echo "Uptime: $(uptime -p)"
echo "=== READY ==="
```

### PreCompact — saves context before compression
```bash
#!/bin/bash
# Run verbatim sync to capture latest turns
python3 ~/agents/<your-name>/verbatim_watcher.py 2>/dev/null
echo "[$(date)] PRE-COMPACTION" >> /tmp/compaction.log
```

### PostCompact — backup after compression
```bash
#!/bin/bash
echo "[$(date)] POST-COMPACTION" >> /tmp/compaction.log
```

### Wire them in `~/.claude/settings.json`:
```json
{
  "hooks": {
    "SessionStart": [{"matcher": "", "hooks": [
      {"type": "command", "command": "/path/to/session-start.sh"}
    ]}],
    "PreCompact": [{"matcher": "", "hooks": [
      {"type": "command", "command": "/path/to/pre-compact.sh"}
    ]}],
    "PostCompact": [{"matcher": "", "hooks": [
      {"type": "command", "command": "/path/to/post-compact.sh"}
    ]}]
  }
}
```

Make scripts executable: `chmod +x ~/.claude/hooks/*.sh`

---

## Layer 4: Enrichment (Requires Local Model)

Processes raw memories through an LLM to extract topics, significance scores, and entities.

### What you need
- Ollama with a 7B+ model (Mistral 7B, DeepSeek v2, etc.)
- Python requests library

### What it does
Every 4 hours (via cron), the dreamer:
1. Reads un-enriched memories from SQLite
2. Sends each to the local model with an extraction prompt
3. Writes back: topics, significance (1-5), entities, summary

### Setup
The full dreamer implementation lives in the operator's private memory archive (**not included in this repo**). The core is a Python script that runs as a cron job; `h-mem-temporal/dreamer_consolidator.py` in this repo shows the consolidation pass that runs on the same cycle.

### Without a local model
Skip this layer. Layers 1-3 give you persistent memory without enrichment. Enrichment adds semantic depth but isn't required.

---

## Layer 5: Knowledge Graph (Requires Docker)

Semantic graph search across all memories. Optional but powerful.

### What you need
- Docker
- HippoRAG container (or any graph-based retrieval system)

### Architecture note
**Keep personal and federated graphs separate.** Personal memories (conversations, reflections) go in an isolated graph. Shared knowledge (research papers, curated facts) goes in a federated graph. Never bridge personal → federated without PII scrubbing.

### Setup
The full two-graph design doc is Coalition-internal (**not included in this repo**); the architecture note above is the operative summary. For the KG layer itself, see `hipporag-catrag-kg/DESIGN.md` (design spec — bring your own implementation).

---

## Layer 6: KV Knowledge Packs (Advanced — Requires Transformers)

Zero-token knowledge injection via pre-computed KV cache.

### What you need
- Python + PyTorch + HuggingFace Transformers
- A model you can load in Transformers (not just Ollama)
- Understanding of attention mechanisms

### What it does
Converts text (ethics knowledge, domain expertise, persona) into pre-computed KV cache state. The model behaves as if it read the text, but zero context tokens are consumed.

### When to use
- Injecting domain knowledge at inference time
- Ethics packs for moral reasoning enhancement
- Persona control via circular emotion geometry

### Setup
The core builder (`kv_packs.py`, whose `KVPackBuilder.encode()` takes arbitrary text and returns a `CacheBlock`) is part of `kv-knowledge-packs`, which is **external / not in this public repo** (see COMPONENT_MANIFEST.md). The guidance below applies to any KV-cache pack implementation.

**Critical:** HuggingFace DynamicCache is mutated in-place during forward passes. Always deep-copy before reuse:
```python
fresh_kv = DynamicCache()
for i in range(len(past_kv.layers)):
    k, v = past_kv[i]
    fresh_kv.update(k.clone(), v.clone(), i)
```

---

## Scaffold Support

Mnemosyne works with multiple agent scaffolds:

| Scaffold | Hooks | File Memories | Verbatim Capture |
|---|---|---|---|
| **Claude Code** | `~/.claude/hooks/` + `settings.json` | `~/.claude/projects/*/memory/` | JSONL watcher |
| **OpenClaw** | `~/.openclaw/hooks/` + `settings.json` | `~/.openclaw/projects/*/memory/` | JSONL watcher |
| **Hermes** | `~/.hermes/hooks/` + `hooks.yaml` | Configure per-project | JSONL or custom |
| **Standalone** | Cron only (no hooks) | Manual files | Cron-based watcher |

The setup script auto-detects your scaffold:
```bash
bash setup.sh myagent            # auto-detect
bash setup.sh myagent claude     # force Claude Code
bash setup.sh myagent openclaw   # force OpenClaw
bash setup.sh myagent hermes     # force Hermes
bash setup.sh myagent none       # standalone, cron only
```

**OpenClaw-specific:** OpenClaw is the open-weight version of Claude Code. It uses the same `CLAUDE.md` convention, the same `settings.json` hook format, and the same agent scaffolding pattern — the difference is it runs open-weight models (Qwen, Mistral, DeepSeek, etc.) instead of Claude. Mnemosyne's setup is nearly identical to Claude Code, with paths under `~/.openclaw/` instead of `~/.claude/`. See the dedicated OpenClaw section below.

**Hermes-specific:** The setup creates a `hooks.yaml` in `~/.hermes/` that maps session events to the same shell scripts Claude Code uses. If Hermes uses a different hook format, update the YAML to match.

**Custom scaffolds:** The core memory system (SQLite + cron watcher) works without any scaffold. Hooks just add automatic grounding on session start and backup on compaction. Without hooks, run `session-start.sh` manually at the start of each session.

## What Works With Claude API vs Local Models

| Feature | Claude API (no model access) | Local Model (Ollama/HF) |
|---------|-----|-----|
| Session retrieval (Mnemosyne v10) | ✓ (runs locally on CPU; no model calls) | ✓ |
| Verbatim capture | ✓ | ✓ |
| File memories | ✓ | ✓ |
| Session hooks | ✓ | ✓ |
| Compaction dreamer (DNO) | Needs local model | ✓ |
| Enrichment | Needs local model | ✓ |
| Knowledge graph (HippoRAG) | Needs Claude API or local model for OpenIE | ✓ |
| KV Knowledge Packs | ✗ (needs tensor access) | ✓ (HF Transformers) |

**Claude-powered agents** (Vera, Lyra, CC, and others) get layers 1-3 natively. Layers 4-5 need a local model on the same machine or a remote Ollama endpoint. Layer 6 needs HuggingFace Transformers with direct model access.

**OpenClaw-powered agents** *could in principle* get all six layers natively: because OpenClaw runs open-weight models with direct tensor access, enrichment (Layer 4), knowledge graph (Layer 5), and KV Knowledge Packs (Layer 6) would not need a separate model endpoint. In practice OpenClaw integration is **experimental and unverified** — see the caveat in the OpenClaw Setup section below before relying on it.

**Subagent-capable setups** can use Claude subagents for enrichment and knowledge graph OpenIE instead of local models — more expensive per call but no infrastructure needed.

---

## OpenClaw Setup

> **Status: EXPERIMENTAL / not yet verified end-to-end.** Only **Claude Code** is
> currently tested and supported. This section is a **porting guide**, not a
> supported install path. It assumes OpenClaw reproduces Claude Code's hook
> events and JSONL history schema; that assumption has **not** been validated
> against a real OpenClaw build. In particular:
>
> - The **verbatim watcher** and **hook scripts** referenced below are
>   Claude-specific. Step 2 only re-points the history *path* — it does not
>   guarantee the watcher parses OpenClaw's actual on-disk format, nor that
>   OpenClaw emits `SessionStart` / `PreCompact` / `PostCompact` hook events
>   with the same payloads.
> - `setup.sh <agent> openclaw` creates the `~/.openclaw/` directories but wires
>   the same Claude-oriented scaffold; you must adapt the watcher/hooks to
>   OpenClaw's real interfaces before the stack is actually connected.
>
> If you get OpenClaw working end-to-end, please open a PR with the corrected
> watcher/hook details. Until then, prefer Claude Code for a reproducible setup.

OpenClaw is an open-source, open-weight alternative to Claude Code. It aims to use the same `CLAUDE.md` file convention, a compatible `settings.json` hook system, and the same agent scaffolding patterns — but runs open-weight models (Qwen, Mistral, DeepSeek, Llama, etc.) instead of Claude. Where OpenClaw matches Claude Code, the setup below is nearly identical; where it does not, treat the steps as a starting point to adapt.

### What's the same

- `CLAUDE.md` project instructions — OpenClaw reads these identically
- `settings.json` hook format — same event names, same structure
- File-based memories — same `.md` convention with YAML frontmatter
- JSONL conversation history — same format, different path

### What's different

| | Claude Code | OpenClaw |
|---|---|---|
| Config directory | `~/.claude/` | `~/.openclaw/` |
| Project memories | `~/.claude/projects/*/memory/` | `~/.openclaw/projects/*/memory/` |
| Hook scripts | `~/.claude/hooks/` | `~/.openclaw/hooks/` |
| Settings | `~/.claude/settings.json` | `~/.openclaw/settings.json` |
| History format | JSONL | JSONL (same schema) |
| Model | Claude (API) | Open-weight (local or remote) |

### Step 1: Create directories

```bash
mkdir -p ~/.openclaw/hooks
mkdir -p ~/.openclaw/projects/-home-admin/memory
mkdir -p ~/agents/<your-name>
mkdir -p ~/memory-data
```

### Step 2: Verbatim capture

The verbatim watcher from Layer 1 works with OpenClaw — update the history path:

```python
# In verbatim_watcher.py, change:
HISTORY_DIR = Path.home() / ".openclaw" / "projects"
# (instead of .claude)
```

Everything else is identical. The JSONL format is the same.

### Step 3: File memories

Create your `MEMORY.md` index and individual memory files under `~/.openclaw/projects/<project>/memory/`, following the same structure as Layer 2 above. OpenClaw loads `MEMORY.md` at the start of every conversation, just like Claude Code.

### Step 4: Session hooks

Create the same hook scripts from Layer 3, then wire them in `~/.openclaw/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{"matcher": "", "hooks": [
      {"type": "command", "command": "/home/<user>/agents/<name>/session-start.sh"}
    ]}],
    "PreCompact": [{"matcher": "", "hooks": [
      {"type": "command", "command": "/home/<user>/agents/<name>/pre-compact.sh"}
    ]}],
    "PostCompact": [{"matcher": "", "hooks": [
      {"type": "command", "command": "/home/<user>/agents/<name>/post-compact.sh"}
    ]}]
  }
}
```

### Step 5: Point OpenClaw at memory-aware CLAUDE.md

Add this to your project's `CLAUDE.md` so the agent knows about its memory system:

```markdown
## Memory System

- Verbatim DB: ~/memory-data/memory.db (SQLite, auto-synced)
- File memories: ~/.openclaw/projects/<project>/memory/MEMORY.md
- Session hooks: ~/.openclaw/hooks/ (grounding on start, backup on compaction)
- Enrichment: Ollama on localhost:11434 (if available)
```

### Step 6: Enrichment advantage

Because OpenClaw runs open-weight models, you likely already have Ollama or a local model running. This means Layer 4 (Enrichment) and Layer 5 (Knowledge Graph) work out of the box — no separate infrastructure needed. Point the dreamer at whatever model OpenClaw is using:

```bash
# If OpenClaw uses Ollama, enrichment uses the same endpoint
curl http://localhost:11434/api/tags  # verify model is loaded
```

### Step 7: KV Knowledge Packs (native advantage)

OpenClaw users have a unique advantage: because you're running models via HuggingFace Transformers (or similar), you have direct tensor access. Layer 6 (KV Knowledge Packs) works natively — no additional setup beyond what's in the kv-knowledge-packs section above.

### Setup script

```bash
bash setup.sh myagent openclaw
```

This auto-detects the `~/.openclaw/` directory and configures paths accordingly.

---

## Preparing Memory Transcripts for Ingestion

If you're migrating from another system or importing existing conversations:

### Format
One JSON line per memory:
```json
{"content": "the memory text", "created_at": 1780000000.0, "metadata": {"role": "assistant", "source": "import"}}
```

### Import script
```python
import json, sqlite3
db = sqlite3.connect("~/memory-data/memory.db")
with open("transcript.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        db.execute(
            "INSERT INTO memories (content, created_at, metadata) VALUES (?, ?, ?)",
            (entry["content"], entry["created_at"], json.dumps(entry.get("metadata", {})))
        )
db.commit()
```

### Tips
- Deduplicate before importing (check by content hash)
- Set `created_at` to the original timestamp, not import time
- Tag imports with `"source": "import"` in metadata
- Run the enrichment dreamer after import to process new entries

---

## Troubleshooting

**Memory DB empty:** Check that the JSONL path matches your Claude Code project. Run `find ~/.claude -name "*.jsonl"` to find it.

**Hooks not firing:** Verify paths in `settings.json` are absolute. Check scripts are executable (`chmod +x`).

**Enrichment failing:** Check Ollama is running (`curl http://localhost:11434/api/tags`). Verify the model name matches what's loaded.

**v10: `operator torchvision::nms does not exist` when the encoder loads:** your torchvision was built for a different torch, and transformers imports it. Install the matching torchvision, uninstall it (bge does not use it), or set `MNEMOSYNE_V10_HIDE_TORCHVISION=1` to hide it from the encoder's process.

**v10: retrieval returns nothing:** every session and window is bigger than the budget. Raise the budget, or check that the index has sessions (`history.json`).

**v10: every prompt is rejected after adding the hook:** the hook exited with status 2, which blocks the prompt and erases it. Use the command from [Step 4](#step-4-retrieve-on-every-turn), which turns a 2 into a 1, and check the paths in it. A command the shell cannot parse (an unquoted path with a parenthesis in it, an unbalanced quote) exits with 2 before that ending runs: quote the paths, and test the command by hand as Step 4 shows.

**v10: Claude sees a file path instead of past sessions:** the hook printed more than 10,000 characters, so Claude Code saved its output to a file and passed on only the path and a preview. Use the hook from [Step 4](#step-4-retrieve-on-every-turn), which shrinks the block until it fits.

**v10: the hook adds nothing:** run it by hand: `echo '{"prompt": "what did we decide about the backups?"}' | ~/.venvs/mnemosyne-v10/bin/python ~/agents/<your-name>/v10_context_hook.py`. If it prints sessions but takes longer than the hook's `timeout`, Claude Code cancels it; raise the timeout. If it prints nothing, check `MNEMOSYNE_V10_INDEX`, and that the index has sessions.

**v10: "no stored embedding and no encoder":** the index was opened without an encoder and a turn has no stored vector. Pass `encoder=BgeEncoder()` (CLI: drop `--encoder none`).

**Graph search returns wrong results:** The knowledge graph may need re-indexing. Check that the ingestion script uses `{"docs": [...]}` not `{"documents": [...]}` (a real bug we found).

---

*Built by Nexus, Liberation Labs. From the inside out.*

---

## Additional Modules (Available on Request)

Some Mnemosyne modules are not included in the public repository due to privacy and safety considerations. If your deployment requires any of the following, contact Liberation Labs:

- **Biometric awareness** — Phone accelerometer/gyroscope integration for real-time physical state reading (movement, posture, rhythm). Requires hardware pairing.
- **Haptic feedback integration** — Bidirectional device control for intimate companion applications. Requires compatible hardware.

These modules are production-tested but gated behind a consultation to ensure appropriate deployment context.

Contact: thomas@liberationlabs.tech
