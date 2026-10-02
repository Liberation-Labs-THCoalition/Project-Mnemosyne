"""Parity with the published v10 run: the library must reproduce the frozen held-out retrieval record.

Needs data that is not in this repository, so every test here skips unless it is provided:

    MNEMOSYNE_V10_DATA     LongMemEval_S-cleaned: longmemeval_s_cleaned.json from xiaowu0162/longmemeval-cleaned
                           (sha256 d6f21ea9...; checked)
    MNEMOSYNE_V10_EMB      the precomputed v10 embeddings: turn_index.json, turns.f16.npy, questions.f16.npy,
                           meta.json (as written by the release's code/embed_turns.py)
    MNEMOSYNE_V10_PROMPTS  optional: the reader prompt files qNNN.txt, to check the rendered history byte for byte

The record, tests/fixtures/recall_heldout_frozen.json, is the release's data/recall_heldout_frozen.json
(sha256 checked). For every one of its 400 held-out questions the library must give the same evidence
coverage, budget tokens, session count and whole-session count. Runtime: several minutes on one CPU core.

Two controls change one parameter each and must be DETECTED, so the comparison is shown to be able to fail.
They run on the record's first CONTROL_N questions only: evaluating an unfrozen configuration on held-out
questions is exactly the peek v10's pre-registration forbids for results, so the controls stay small and
assert only that differences exist. Their totals are not results; do not report them.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from mnemosyne_v10 import frozen_config, make_config  # noqa: E402
from mnemosyne_v10 import longmemeval as L  # noqa: E402

RECORD = os.path.join(HERE, "fixtures", "recall_heldout_frozen.json")
CONTROL_N = 50
DATA = os.environ.get("MNEMOSYNE_V10_DATA")
EMB = os.environ.get("MNEMOSYNE_V10_EMB")
PROMPTS = os.environ.get("MNEMOSYNE_V10_PROMPTS")

pytestmark = pytest.mark.skipif(
    not (DATA and EMB and os.path.isfile(DATA) and os.path.isfile(os.path.join(EMB or "", "turns.f16.npy"))),
    reason="set MNEMOSYNE_V10_DATA and MNEMOSYNE_V10_EMB to run the parity test")


@pytest.fixture(scope="module")
def env():
    assert L.sha256_file(RECORD) == L.HELDOUT_RECORD_SHA256, "fixture is not the published record"
    rec = L.load_record(RECORD)
    data = L.load_dataset(DATA)
    emb = L.Embeddings(EMB, data_sha256=L.DATA_SHA256)
    return rec, data, emb


_rows = {}


def rows_for(env, **overrides):
    rec, data, emb = env
    key = tuple(sorted(overrides.items()))
    if key not in _rows:
        cfg = make_config(overrides)
        idxs = [r["idx"] for r in rec["rows"]][:None if not overrides else CONTROL_N]
        _rows[key] = L.evaluate(data, idxs, cfg, emb, keep_text=bool(PROMPTS) and not overrides)
    return _rows[key]


def test_record_is_the_frozen_config(env):
    rec, _, _ = env
    assert rec["cfg"] == frozen_config() and rec["hash"] == "7bc75f509aa0"
    assert len(rec["rows"]) == 400


def test_heldout_parity(env):
    rec, _, _ = env
    rows = rows_for(env)
    diffs = L.compare(rows, rec["rows"])
    assert diffs == [], f"{len({d[0] for d in diffs})} questions differ, first: {diffs[:5]}"
    s = L.summarise(rows)
    assert (s["n_scored"], s["all_sessions"]) == (376, 365)            # 97.1%, has_answer sessions
    assert s["official_all_sessions"] == 361                           # 96.0%, answer_session_ids


@pytest.mark.parametrize("overrides", [{"w_dense": 1.0}, {"budget": 8500}], ids=["w_dense=1", "budget=8500"])
def test_parity_check_detects_a_changed_parameter(env, overrides):
    rec, _, _ = env
    rows = rows_for(env, **overrides)
    assert len(rows) == CONTROL_N
    diffs = L.compare(rows, rec["rows"][:CONTROL_N])
    assert diffs, f"changing {overrides} went undetected: the parity check cannot fail"


@pytest.mark.skipif(os.environ.get("MNEMOSYNE_V10_ENCODER_CHECK") != "1",
                    reason="set MNEMOSYNE_V10_ENCODER_CHECK=1 to load bge on CPU and embed a small sample")
def test_bge_encoder_reproduces_stored_vectors(env):
    """A sample only (8 user turns, 2 questions), on CPU: never the whole corpus."""
    import numpy as np
    from mnemosyne_v10 import BgeEncoder, turn_key
    rec, data, emb = env
    enc = BgeEncoder(device="cpu", threads=2)
    qidx = [rec["rows"][0]["idx"], rec["rows"][-1]["idx"]]
    texts = [t["content"] for t in data[qidx[0]]["haystack_sessions"][0] if t["role"] == "user"][:4]
    texts += [t["content"] for t in data[qidx[1]]["haystack_sessions"][-1] if t["role"] == "user"][:4]
    stored = emb.store.vectors(emb.store.rows([turn_key("user", t) for t in texts])).astype(np.float32)
    got = enc.embed_turns(texts)
    assert (np.sum(got * stored, axis=1) > 0.999).all()
    q = np.stack([enc.embed_query(data[i]["question"]) for i in qidx])
    assert (np.sum(q * emb.questions[qidx].astype(np.float32), axis=1) > 0.999).all()


@pytest.mark.skipif(not PROMPTS, reason="set MNEMOSYNE_V10_PROMPTS to compare with the reader's prompt files")
def test_rendered_history_is_what_the_reader_saw(env):
    bad = []
    rows = rows_for(env)
    assert len(rows) == 400 and all("text" in r for r in rows)
    for row in rows:
        with open(os.path.join(PROMPTS, f"q{row['idx']:03d}.txt"), encoding="utf-8") as f:
            prompt = f.read()
        start = prompt.index("History Chats:\n\n") + len("History Chats:\n\n")
        end = prompt.rindex("\n\nCurrent Date: ")
        if prompt[start:end] != row["text"]:
            bad.append(row["idx"])
    assert bad == [], f"{len(bad)} prompts differ, e.g. {bad[:5]}"
