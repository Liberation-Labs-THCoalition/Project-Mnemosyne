#!/usr/bin/env bash
#
# Project Mnemosyne — minimal, idempotent setup for the core memory scaffold.
#
# AGENT_SETUP.md refers to this script. It provisions the *core* memory system
# (memory data dir + scaffold config/memory dirs + optional Python venv) and
# prints an ordered checklist for the remaining, per-component steps. It is
# intentionally conservative: it never overwrites existing files, never sends
# notifications, and never contacts the network except for an opt-in venv/pip
# step you trigger with --venv.
#
# It does NOT install the full multi-service stack — several layers are external
# or design-only (see COMPONENT_MANIFEST.md). Per-component install lives in each
# component's own README.
#
# Usage:
#   bash setup.sh <agent_name> [scaffold]
#     scaffold = auto (default) | claude | openclaw | hermes | none
#   bash setup.sh <agent_name> [scaffold] --venv    # also create ./.venv
#
# Examples:
#   bash setup.sh myagent            # auto-detect scaffold
#   bash setup.sh myagent claude     # force Claude Code
#   bash setup.sh myagent openclaw   # force OpenClaw (experimental — see AGENT_SETUP.md)
#   bash setup.sh myagent none       # standalone, cron only
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()  { printf '  %s\n' "$*"; }
head() { printf '\n== %s ==\n' "$*"; }

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

AGENT_NAME="$1"; shift || true
SCAFFOLD="auto"
MAKE_VENV="0"
for arg in "$@"; do
  case "$arg" in
    claude|openclaw|hermes|none|auto) SCAFFOLD="$arg" ;;
    --venv) MAKE_VENV="1" ;;
    *) log "WARN: ignoring unknown argument '$arg'" ;;
  esac
done

head "Detecting scaffold"
if [[ "$SCAFFOLD" == "auto" ]]; then
  if   [[ -d "$HOME/.claude"   ]]; then SCAFFOLD="claude"
  elif [[ -d "$HOME/.openclaw" ]]; then SCAFFOLD="openclaw"
  elif [[ -d "$HOME/.hermes"   ]]; then SCAFFOLD="hermes"
  else SCAFFOLD="none"; fi
  log "auto-detected: $SCAFFOLD"
else
  log "forced: $SCAFFOLD"
fi

case "$SCAFFOLD" in
  claude)   CFG_DIR="$HOME/.claude" ;;
  openclaw) CFG_DIR="$HOME/.openclaw" ;;
  hermes)   CFG_DIR="$HOME/.hermes" ;;
  none)     CFG_DIR="" ;;
esac

head "Core memory directories (idempotent)"
MEMORY_DATA="${MNEMOSYNE_MEMORY_DATA:-$HOME/memory-data}"
mkdir -p "$MEMORY_DATA"
log "memory data dir: $MEMORY_DATA (SQLite store goes here: memory.db)"

if [[ -n "$CFG_DIR" ]]; then
  mkdir -p "$CFG_DIR/projects/$AGENT_NAME/memory"
  mkdir -p "$CFG_DIR/hooks"
  log "scaffold config dir: $CFG_DIR"
  log "file-memory dir:     $CFG_DIR/projects/$AGENT_NAME/memory"
  log "hooks dir:           $CFG_DIR/hooks  (wire per AGENT_SETUP.md 'Session Hooks')"
else
  log "standalone mode: no scaffold config dir. Use cron + manual memory files."
fi

if [[ "$MAKE_VENV" == "1" ]]; then
  head "Python venv (opt-in)"
  if [[ ! -d "$REPO_ROOT/.venv" ]]; then
    python3 -m venv "$REPO_ROOT/.venv"
    log "created $REPO_ROOT/.venv"
  else
    log "$REPO_ROOT/.venv already exists — leaving it"
  fi
  log "activate with: source $REPO_ROOT/.venv/bin/activate"
fi

head "Next steps (see COMPONENT_MANIFEST.md for what is implemented vs external)"
cat <<EOF
  1. Verbatim capture (Layer 1) and session hooks (Layer 3): AGENT_SETUP.md.
  2. Per-component installs (each has its own README):
       - kintsugi-cma         (docker compose up  — build context fixed to '.')
       - h-mem-temporal, sira-enrichment, tgs-verification, tgs-rag-bridge
       - dispatch-notion-memory  (run/test with PYTHONPATH=src)
       - metacognition           (pip install -r requirements.txt; jlens from GitHub)
       - swarm                   (cp swarm/.env.example swarm/.env; NATS/Ollama)
  3. External / not in this public repo: oracle-memory, kv-knowledge-packs,
     and the mnemosyne-metacognition standalone repo. See COMPONENT_MANIFEST.md.
  4. Design-only specs (no runnable service yet): hipporag-catrag-kg,
     mnemosyne-wiki, routines.

Scaffold: $SCAFFOLD   Agent: $AGENT_NAME   Repo: $REPO_ROOT
Done. Re-running this script is safe (idempotent).
EOF
