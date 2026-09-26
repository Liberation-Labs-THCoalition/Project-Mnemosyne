"""LongMemEval_S-cleaned evaluation of the library: evidence recall, and parity with the frozen record.

Retrieval only ever sees ``view(item)``: the question, the question date, and each session's date and
(role, content) turns. Labels (``has_answer``, ``answer_session_ids``, ids, types) are read only here, for
scoring, after retrieval. This mirrors v10.py's ``view()`` / ``evidence()`` split.
"""
import hashlib
import json
import os
import statistics

import numpy as np

from . import encoders as _enc
from .config import config_hash
from .retrieval import MemoryIndex

# sha256 of LongMemEval_S-cleaned (xiaowu0162/longmemeval-cleaned, longmemeval_s_cleaned.json), the file the
# published run used; the frozen record is only comparable on exactly this file.
DATA_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
# sha256 of the published held-out retrieval record (data/recall_heldout_frozen.json in the paper release).
HELDOUT_RECORD_SHA256 = "a9717a27aaf7a5f7d15e250cf54b11ab35039256a2c38f323f845da3e148a965"

# Per-question fields that must match the frozen record. (The record's ``tm_all_sessions_strict`` is v10.py's
# like-for-like check against the v9 matcher on its own debug render; it is not a property of the retriever.)
PARITY_FIELDS = ("qid", "type", "abstention", "n_ev", "covered", "all_sessions", "all_turns",
                 "tokens", "n_sessions", "whole", "time_windows")


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def view(item):
    """The ONLY input retrieval gets. Built field by field; no ids, no labels, no type."""
    v = {"question": str(item["question"]), "question_date": str(item["question_date"]),
         "sessions": [{"date": str(d),
                       "turns": [{"role": str(t["role"]), "content": str(t["content"])} for t in s]}
                      for d, s in zip(item["haystack_dates"], item["haystack_sessions"])]}
    assert set(v) == {"question", "question_date", "sessions"}
    assert all(set(s) == {"date", "turns"} and all(set(t) == {"role", "content"} for t in s["turns"])
               for s in v["sessions"])
    return v


def evidence(item):
    """Scoring only: {session index: [indexes of its has_answer turns]}."""
    ev = {}
    for si, s in enumerate(item["haystack_sessions"]):
        ts = [ti for ti, t in enumerate(s) if t.get("has_answer")]
        if ts:
            ev[si] = ts
    return ev


def official_sessions(item):
    """Scoring only: indexes of the sessions listed in the official ``answer_session_ids``."""
    want = set(item["answer_session_ids"])
    return [si for si, sid in enumerate(item["haystack_session_ids"]) if sid in want]


class Embeddings:
    """The published run's precomputed embeddings: the turn store plus ``questions.f16.npy`` (one row per
    question, in dataset order)."""

    def __init__(self, directory, data_sha256=None):
        self.store = _enc.EmbeddingStore.open(directory)
        meta = self.store.meta or {}
        if _enc.space_of(meta) != _enc.space_of(_enc.V10_META):
            raise ValueError(f"{directory}: embeddings are not v10's ({_enc.space_of(meta)})")
        if data_sha256 and meta.get("data_sha256") not in (None, data_sha256):
            raise ValueError(f"{directory}: embeddings were computed on a different dataset file")
        self.questions = np.load(os.path.join(directory, "questions.f16.npy"))


def load_dataset(path, check_sha256=True):
    if check_sha256:
        got = sha256_file(path)
        if got != DATA_SHA256:
            raise ValueError(f"{path}: sha256 {got[:16]}..., expected LongMemEval_S-cleaned {DATA_SHA256[:16]}...")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def retrieve_item(item, qidx, cfg, emb):
    v = view(item)
    index = MemoryIndex(config=cfg, embeddings=emb.store if "dense" in cfg["channels"] else None)
    index.add_sessions(v["sessions"])
    qvec = emb.questions[qidx] if "dense" in cfg["channels"] else None
    return v, index.retrieve(v["question"], question_date=v["question_date"], query_embedding=qvec)


def evaluate(data, idxs, cfg, emb, keep_text=False):
    """One row per question, with the fields of v10.py's retrieval record plus the official-sessions metric."""
    rows = []
    for qi in idxs:
        item = data[qi]
        v, r = retrieve_item(item, qi, cfg, emb)
        chosen = r.chosen
        ev = evidence(item)
        inc = lambda si, ti: si in chosen and (chosen[si] is None or ti in chosen[si])
        cov = {si: any(inc(si, ti) for ti in tis) for si, tis in ev.items()}
        off = official_sessions(item)
        row = dict(idx=qi, qid=item["question_id"], type=item["question_type"],
                   abstention=item["question_id"].endswith("_abs"), n_ev=len(ev),
                   covered=sum(cov.values()),
                   all_sessions=bool(ev) and all(cov.values()),
                   all_turns=bool(ev) and all(inc(si, ti) for si, tis in ev.items() for ti in tis),
                   tokens=r.tokens, n_sessions=r.n_sessions, whole=r.whole, time_windows=r.time_windows,
                   official_all_sessions=bool(off) and all(si in chosen for si in off))
        if keep_text:
            row["text"] = r.text
        rows.append(row)
    return rows


def summarise(rows):
    scored = [r for r in rows if not r["abstention"] and r["n_ev"] > 0]
    return {"n": len(rows), "n_scored": len(scored),
            "all_sessions": sum(r["all_sessions"] for r in scored),
            "official_all_sessions": sum(r["official_all_sessions"] for r in scored),
            "median_tokens": float(statistics.median(r["tokens"] for r in rows)) if rows else 0.0}


def load_record(path):
    """The frozen record: {"cfg", "hash", "summary", "rows"} (v10.py's run output, first result)."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    rec = doc[0] if isinstance(doc, list) else doc
    if config_hash(rec["cfg"]) != rec["hash"]:
        raise ValueError(f"{path}: the recorded config does not hash to {rec['hash']}")
    return rec


def compare(rows, record_rows, fields=PARITY_FIELDS):
    """Differences between our rows and the record's, per question: [(idx, field, ours, record)]."""
    ours = {r["idx"]: r for r in rows}
    diffs = []
    for rec in record_rows:
        mine = ours.get(rec["idx"])
        if mine is None:
            diffs.append((rec["idx"], "<missing>", None, rec.get("qid")))
            continue
        for f in fields:
            if mine[f] != rec[f]:
                diffs.append((rec["idx"], f, mine[f], rec[f]))
    extra = set(ours) - {rec["idx"] for rec in record_rows}
    diffs += [(i, "<extra>", ours[i].get("qid"), None) for i in sorted(extra)]
    return diffs
