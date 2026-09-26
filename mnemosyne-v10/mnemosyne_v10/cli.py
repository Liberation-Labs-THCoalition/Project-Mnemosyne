"""Command line: build an index, add sessions, retrieve, and check parity on LongMemEval_S-cleaned.

    python -m mnemosyne_v10 index    --history history.json --out ./v10-index
    python -m mnemosyne_v10 add      --index ./v10-index --session session.json
    python -m mnemosyne_v10 retrieve --index ./v10-index --question "What did I say about ...?"
    python -m mnemosyne_v10 longmemeval --data longmemeval_s_cleaned.json --embeddings EMB_DIR \\
        --record tests/fixtures/recall_heldout_frozen.json

Every path is an argument. Defaults are the frozen v10 configuration; ``--config FILE`` and
``--set KEY=VALUE`` (VALUE parsed as JSON, e.g. ``--set 'channels=["bm25"]'``) override it.
"""
import argparse
import json
import sys
import time

from .config import config_hash, load_config


def _overrides(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set expects KEY=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _config(a):
    return load_config(a.config, _overrides(a.set))


def _encoder(a, cfg):
    if "dense" not in cfg["channels"] or a.encoder == "none":
        return None
    from .encoders import BgeEncoder
    return BgeEncoder(device=a.device, threads=a.threads, local_files_only=a.local_files_only)


def _read_sessions(path):
    with (sys.stdin if path == "-" else open(path, encoding="utf-8")) as f:
        doc = json.load(f)
    if isinstance(doc, dict) and "turns" in doc:
        doc = [doc]
    return doc["sessions"] if isinstance(doc, dict) else doc


def cmd_index(a):
    from .encoders import EmbeddingStore
    from .retrieval import MemoryIndex
    cfg = _config(a)
    store = EmbeddingStore.open(a.embeddings) if a.embeddings else None
    index = MemoryIndex(config=cfg, encoder=_encoder(a, cfg), embeddings=store)
    t0 = time.time()
    index.add_sessions(_read_sessions(a.history))
    index.save(a.out)
    print(f"indexed {len(index)} sessions / {index.n_turns} turns in {time.time() - t0:.1f}s -> {a.out}",
          file=sys.stderr)


def cmd_add(a):
    from .retrieval import MemoryIndex
    cfg = _config(a)
    index = MemoryIndex.load(a.index, config=cfg, encoder=_encoder(a, cfg))
    new = index.add_sessions(_read_sessions(a.session))
    index.save(a.index)
    print(f"added {len(new)} session(s); index now {len(index)} sessions / {index.n_turns} turns",
          file=sys.stderr)


def cmd_retrieve(a):
    from .retrieval import MemoryIndex
    cfg = _config(a)
    question = sys.stdin.read() if a.question == "-" else a.question
    index = MemoryIndex.load(a.index, config=cfg, encoder=_encoder(a, cfg))
    r = index.retrieve(question, question_date=a.question_date, budget=a.budget)
    if a.format == "json":
        json.dump({"config_hash": config_hash(cfg if a.budget is None else {**cfg, "budget": a.budget}),
                   "tokens": r.tokens, "budget": r.budget, "n_sessions": r.n_sessions, "whole": r.whole,
                   "sessions": [{"index": s.index, "date": s.date, "whole": s.whole,
                                 "turn_indexes": list(s.turn_indexes)} for s in r.sessions],
                   "text": r.render(ensure_ascii=not a.unescaped)}, sys.stdout, ensure_ascii=False)
        print()
    else:
        sys.stdout.write(r.render(ensure_ascii=not a.unescaped))


def cmd_longmemeval(a):
    from . import longmemeval as L
    cfg = _config(a)
    t0 = time.time()
    data = L.load_dataset(a.data, check_sha256=not a.no_sha_check)
    emb = L.Embeddings(a.embeddings, data_sha256=L.DATA_SHA256)
    rec = L.load_record(a.record) if a.record else None
    if rec is not None:
        idxs = [r["idx"] for r in rec["rows"]]
    elif a.idxs:
        idxs = [int(x) for x in a.idxs.split(",")]
    else:
        idxs = list(range(len(data)))
    if a.limit:
        idxs = idxs[:a.limit]
    print(f"config {config_hash(cfg)}: {json.dumps(cfg, sort_keys=True)}", flush=True)
    if rec is not None and config_hash(cfg) != rec["hash"]:
        print(f"note: config {config_hash(cfg)} is not the record's frozen config {rec['hash']}. On held-out "
              "questions this is a control, not a result: do not report its totals.", flush=True)
    rows = L.evaluate(data, idxs, cfg, emb)
    s = L.summarise(rows)
    print(f"{s['n']} questions, {s['n_scored']} scored: all has_answer sessions {s['all_sessions']}/{s['n_scored']}"
          f" = {s['all_sessions'] / max(s['n_scored'], 1):.4f}; all answer_session_ids sessions "
          f"{s['official_all_sessions']}/{s['n_scored']} = {s['official_all_sessions'] / max(s['n_scored'], 1):.4f};"
          f" median budget tokens {s['median_tokens']:.1f} ({time.time() - t0:.0f}s)", flush=True)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"cfg": cfg, "hash": config_hash(cfg), "summary": s, "rows": rows}, f)
    if rec is None:
        return 0
    wanted = set(idxs)
    rec_rows = [r for r in rec["rows"] if r["idx"] in wanted]
    diffs = L.compare(rows, rec_rows)
    qs = sorted({d[0] for d in diffs})
    fields = sorted({d[1] for d in diffs})
    print(f"parity with {a.record} (record config {rec['hash']}): {len(qs)} of {len(rec_rows)} questions differ"
          + (f" in {fields}" if diffs else ""))
    for d in diffs[:a.show]:
        print(f"  idx {d[0]}: {d[1]} ours={d[2]} record={d[3]}")
    return 1 if diffs else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mnemosyne-v10", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, encoder=True):
        p.add_argument("--config", help="JSON file of config overrides (default: frozen v10)")
        p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override one config key")
        if encoder:
            p.add_argument("--encoder", choices=["bge", "none"], default="bge",
                           help="dense encoder for new turns / the question (default bge)")
            p.add_argument("--device", default="cpu", help="torch device for the encoder (default cpu)")
            p.add_argument("--threads", type=int, default=None, help="torch CPU threads")
            p.add_argument("--local-files-only", action="store_true", help="never download the encoder")

    p = sub.add_parser("index", help="build an index from a history file")
    p.add_argument("--history", required=True, help='JSON: {"sessions": [{"date", "turns"}]} or a list; - = stdin')
    p.add_argument("--out", required=True, help="index directory to write")
    p.add_argument("--embeddings", help="existing embedding directory to reuse vectors from (read only)")
    common(p)
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("add", help="append sessions to an index")
    p.add_argument("--index", required=True)
    p.add_argument("--session", required=True, help="JSON: one {date, turns} session, a list, or {sessions}; - = stdin")
    common(p)
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser("retrieve", help="retrieve context for a question")
    p.add_argument("--index", required=True)
    p.add_argument("--question", required=True, help="question text; - reads it from stdin")
    p.add_argument("--question-date", help="needed only by the time channel")
    p.add_argument("--budget", type=int, help="token budget for this call (default: config budget, 24000)")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("--unescaped", action="store_true",
                   help="render non-ASCII unescaped (fewer tokens; the benchmark reader saw it escaped)")
    common(p)
    p.set_defaults(fn=cmd_retrieve)

    p = sub.add_parser("longmemeval", help="evidence recall on LongMemEval_S-cleaned, and parity with a record")
    p.add_argument("--data", required=True, help="longmemeval_s_cleaned.json")
    p.add_argument("--embeddings", required=True, help="precomputed v10 embeddings (turn_index.json, "
                                                       "turns.f16.npy, questions.f16.npy, meta.json)")
    p.add_argument("--record", help="frozen retrieval record to compare with (questions are taken from it)")
    p.add_argument("--idxs", help="comma-separated dataset indexes (without --record)")
    p.add_argument("--limit", type=int)
    p.add_argument("--out", help="write our rows as JSON")
    p.add_argument("--show", type=int, default=20, help="differences to print")
    p.add_argument("--no-sha-check", action="store_true", help="skip the dataset sha256 check")
    common(p, encoder=False)
    p.set_defaults(fn=cmd_longmemeval)

    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
