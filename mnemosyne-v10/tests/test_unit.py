"""Self-contained tests on a synthetic history. No dataset, no torch, no network.

The dense encoder is a deterministic stub and the budget uses a whitespace token counter, so these run
anywhere numpy and nltk are installed. The tests that need the real o200k tokenizer, scikit-learn or torch
skip when those are unavailable.
"""
import collections
import datetime
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from mnemosyne_v10 import (FROZEN_V10_FILE_SHA256, FROZEN_V10_HASH, MemoryIndex, config_hash,  # noqa: E402
                           frozen_config, make_config, render_history, terms, turn_key)
from mnemosyne_v10 import retrieval as R  # noqa: E402
from mnemosyne_v10.config import frozen_file_bytes  # noqa: E402


class StubEncoder:
    """Hashed bag of words, L2-normalised. Records what it was asked to embed."""
    meta = {"model": "stub", "revision": "0", "pooling": "bow", "max_length": 0, "query_prefix": ""}
    DIM = 64

    def __init__(self):
        self.turn_texts, self.queries = [], []

    def _vec(self, text):
        v = np.zeros(self.DIM, dtype=np.float32)
        for w in re.findall(r"[a-z]+", text.lower()):
            v[hashlib.md5(w.encode()).digest()[0] % self.DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_turns(self, texts):
        texts = list(texts)
        self.turn_texts += texts
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.DIM), np.float32)

    def embed_query(self, text):
        self.queries.append(text)
        return self._vec(text)


def words(text):
    return len(text.split())


def U(c):
    return {"role": "user", "content": c}


def A(c):
    return {"role": "assistant", "content": c}


FILLER = ["I want to plan a trip to the mountains next spring.", "Here are some hiking trails you could try.",
          "Can you suggest a recipe for dinner tonight?", "A lentil soup is quick and cheap to make.",
          "What is a good way to learn Spanish?", "Daily practice with short lessons works well.",
          "My laptop keeps overheating when I play games.", "Clean the fans and check the thermal paste."]


def history():
    """Seven sessions, deliberately NOT in date order. Session 4 holds the fact asked about."""
    sessions = []
    dates = ["2023/05/03 (Wed) 10:00", "2023/05/01 (Mon) 09:00", "2023/05/07 (Sun) 18:30",
             "2023/05/02 (Tue) 12:00", "2023/05/06 (Sat) 08:15", "2023/05/05 (Fri) 20:00",
             "2023/05/04 (Thu) 14:45"]
    for i, d in enumerate(dates):
        turns = []
        for j in range(4):
            turns.append(U(FILLER[(i + 2 * j) % len(FILLER)] + f" (note {i}.{j})"))
            turns.append(A(FILLER[(i + 2 * j + 1) % len(FILLER)] + f" (reply {i}.{j})"))
        sessions.append({"date": d, "turns": turns})
    sessions[3]["turns"][4] = U("My sister Beatrice just adopted a grey greyhound called Pixel.")
    sessions[3]["turns"][5] = A("Congratulations to Beatrice! Greyhounds are gentle dogs.")
    return sessions


QUESTION = "What is the name of the greyhound my sister adopted?"


def build(config=None, sessions=None, encoder=None):
    enc = encoder if encoder is not None else StubEncoder()
    idx = MemoryIndex(config=config, encoder=enc, token_counter=words)
    idx.add_sessions(sessions if sessions is not None else history())
    return idx, enc


# ---------------------------------------------------------------- the frozen configuration

def test_frozen_defaults_are_the_published_config():
    assert hashlib.sha256(frozen_file_bytes()).hexdigest() == FROZEN_V10_FILE_SHA256
    cfg = frozen_config()
    assert config_hash(cfg) == FROZEN_V10_HASH == "7bc75f509aa0"
    assert cfg["channels"] == ["bm25", "dense"]
    assert (cfg["rrf_k"], cfg["w_bm25"], cfg["w_dense"], cfg["second_weight"]) == (60, 1.0, 2.0, 1.0)
    assert (cfg["budget"], cfg["window"], cfg["window_seeds"], cfg["whole_top"]) == (24000, 2, 2, 99)
    assert (cfg["bm25_k1"], cfg["bm25_b"]) == (1.2, 0.75)
    assert MemoryIndex(config={"channels": ["bm25"]}, token_counter=words).config["budget"] == 24000


def test_config_rejects_typos():
    with pytest.raises(ValueError):
        make_config({"w_dnese": 1.0})
    with pytest.raises(ValueError):
        make_config({"channels": ["bm25", "sparse"]})
    with pytest.raises(ValueError):
        make_config({"budget": "24k"})
    with pytest.raises(ValueError):
        make_config({"budget": True})                                  # a bool is an int to Python


# ---------------------------------------------------------------- retrieval behaviour

def test_relevant_session_whole_and_chronological():
    idx, enc = build()
    r = idx.retrieve(QUESTION)
    assert r.tokens <= r.budget == 24000
    assert 3 in r.chosen and r.chosen[3] is None                     # the fact's session, whole
    assert r.n_sessions == 7 and r.whole == 7                         # a big budget takes everything
    assert [s.date for s in r.sessions] == sorted(s["date"] for s in history())
    assert r.text.count("\n### Session ") == 7
    assert enc.queries == [QUESTION]


def test_score_order_decides_what_fits():
    idx, _ = build()
    one = sum(words(json.dumps(t, ensure_ascii=False)) + 1 for t in history()[3]["turns"]) + R.SESSION_HEADER_TOKENS
    r = idx.retrieve(QUESTION, budget=one)
    assert list(r.chosen) == [3] and r.tokens == one                 # exactly one session fits: the right one


def test_window_fallback_when_the_session_does_not_fit():
    idx, _ = build()
    whole = sum(words(json.dumps(t, ensure_ascii=False)) + 1 for t in history()[3]["turns"]) + R.SESSION_HEADER_TOKENS
    r = idx.retrieve(QUESTION, budget=whole - 1)
    assert r.tokens <= whole - 1
    sel = r.chosen[3]
    assert sel is not None and 4 in sel                               # a window around the evidence turn
    assert sel == sorted(sel) and set(sel) <= set(range(8)) and len(sel) < 8
    # windows are +/-2 turns around the session's two best turns
    tops = sorted(((f, ti) for ti, f in enumerate(_fused(idx, QUESTION)[idx._session_start[3]:][:8])),
                  reverse=True)[:2]
    want = sorted({j for _, ti in tops for j in range(max(0, ti - 2), min(8, ti + 3))})
    assert sel == want


def _fused(idx, question):
    cfg = idx.config
    tb = idx._tiebreak()
    f = cfg["w_bm25"] * R.rrf(idx.bm25_scores(question, cfg["bm25_k1"], cfg["bm25_b"]), cfg["rrf_k"], True, tb)
    f = f + cfg["w_dense"] * R.rrf(idx.dense_scores(idx.encoder.embed_query(question)), cfg["rrf_k"], False, tb)
    return f


def test_dense_channel_embeds_user_turns_only():
    idx, enc = build()
    users = [t["content"] for s in history() for t in s["turns"] if t["role"] == "user"]
    assert sorted(enc.turn_texts) == sorted(set(users))
    assert not any("reply" in t or "Congratulations" in t for t in enc.turn_texts)
    assert np.isnan(idx.dense_scores(enc.embed_query(QUESTION))[[p for p in range(idx.n_turns)
                                                                   if idx._assistant[p]]]).all()


def test_labels_and_extra_keys_never_reach_retrieval():
    plain, _ = build()
    labelled = history()
    for s in labelled:
        for t in s["turns"]:
            t["has_answer"] = "Beatrice" in t["content"]
            t["session_id"] = "answer_deadbeef"
    tagged, _ = build(sessions=labelled)
    a, b = plain.retrieve(QUESTION, budget=120), tagged.retrieve(QUESTION, budget=120)
    assert a.chosen == b.chosen and a.tokens == b.tokens and a.text == b.text
    assert "has_answer" not in b.text and "answer_deadbeef" not in b.text


def test_selection_does_not_depend_on_session_order():
    base, _ = build()
    want = base.retrieve(QUESTION, budget=150)
    content = lambda idx, r: sorted((idx.sessions[si][0], json.dumps(v)) for si, v in r.chosen.items())
    for seed in range(3):
        shuffled = history()
        random.Random(seed).shuffle(shuffled)
        idx, _ = build(sessions=shuffled)
        got = idx.retrieve(QUESTION, budget=150)
        assert content(idx, got) == content(base, want) and got.tokens == want.tokens and got.text == want.text


def test_bm25_only_needs_no_encoder():
    idx = MemoryIndex(config={"channels": ["bm25"]}, token_counter=words)
    idx.add_sessions(history())
    r = idx.retrieve(QUESTION)
    assert list(r.chosen)[0] == 3 and r.chosen[3] is None


def test_dense_needs_an_encoder_or_store():
    with pytest.raises(ValueError):
        MemoryIndex(token_counter=words).add_sessions(history())
    idx, _ = build()
    idx.encoder = None
    with pytest.raises(ValueError):
        idx.retrieve(QUESTION)


def test_empty_index():
    idx, _ = build(sessions=[])
    r = idx.retrieve(QUESTION)
    assert r.chosen == {} and r.tokens == 0 and r.text == ""


def test_time_channel_boosts_sessions_in_the_window():
    idx, _ = build(config={"channels": ["bm25", "time"]})
    r = idx.retrieve("What did I cook two days ago?", question_date="2023/05/08 (Mon) 09:00")
    assert r.time_windows == 1
    # no BM25 match anywhere; the window is 05/05-05/07 and 05/06 (session 4) is nearest its centre
    assert list(r.chosen)[:3] == [4] + sorted([2, 5], key=lambda si: idx._session_hash[si])
    with pytest.raises(ValueError):
        idx.retrieve("What did I cook two days ago?")                # the time channel needs a date


def test_time_windows_are_v10s():
    d = datetime.datetime
    cases = {                                                        # question date: Wednesday 2023/05/10
        "What did I do last week?": [(d(2023, 4, 26), d(2023, 5, 10))],
        "Where did we go last weekend?": [(d(2023, 5, 1), d(2023, 5, 10))],
        "What did I buy last month?": [(d(2023, 3, 9), d(2023, 5, 10))],
        "Who called yesterday?": [(d(2023, 5, 8), d(2023, 5, 10))],
        "What did I cook two days ago?": [(d(2023, 5, 7), d(2023, 5, 9))],
        "What happened three weeks ago?": [(d(2023, 4, 15, 12), d(2023, 4, 22, 12))],
        "What did I do last Friday?": [(d(2023, 5, 4), d(2023, 5, 6))],
        "What did I read in March?": [(d(2023, 3, 1), d(2023, 3, 31))],
        "What have I planted since April?": [(d(2023, 4, 1), d(2023, 5, 10))],
        "Anything new this week?": [(d(2023, 5, 3), d(2023, 5, 10))],
        "What is my sister's dog called?": [],
    }
    for question, want in cases.items():
        assert R.time_windows(question, "2023/05/10 (Wed) 12:00") == want, question


def test_save_and_load_round_trip(tmp_path):
    idx, enc = build()
    before = idx.retrieve(QUESTION, budget=150)
    idx.save(tmp_path / "index")
    enc2 = StubEncoder()
    again = MemoryIndex.load(tmp_path / "index", encoder=enc2, token_counter=words)
    assert enc2.turn_texts == []                                      # vectors came from disk
    after = again.retrieve(QUESTION, budget=150)
    assert after.chosen == before.chosen and after.text == before.text
    again.add_session([U("Pixel the greyhound loves the beach."), A("That sounds lovely.")],
                      date="2023/05/09 (Tue) 11:00")
    assert enc2.turn_texts == ["Pixel the greyhound loves the beach."]  # only the new user turn is embedded
    # what `mnemosyne-v10 add` does: vectors from the memory-mapped store and new ones, saved in place
    mixed = again.retrieve("Where does Pixel like to go?")
    again.save(tmp_path / "index")
    third = MemoryIndex.load(tmp_path / "index", encoder=StubEncoder(), token_counter=words)
    assert third.retrieve("Where does Pixel like to go?").text == mixed.text
    assert np.array_equal(third._dense_matrix(), again._dense_matrix())


def test_store_refuses_a_different_embedding_space(tmp_path):
    idx, _ = build()
    idx.save(tmp_path / "index")

    class Other(StubEncoder):
        meta = dict(StubEncoder.meta, model="other")
    with pytest.raises(ValueError):
        MemoryIndex.load(tmp_path / "index", encoder=Other(), token_counter=words)


# ---------------------------------------------------------------- scoring details that decide close calls

def bm25_index(sessions):
    idx = MemoryIndex(config={"channels": ["bm25"]}, token_counter=words)
    idx.add_sessions(sessions)
    return idx


FILL = [{"date": f"2023/06/0{i} (Thu) 10:00", "turns": [U(f"filler talk number {w}"), A(f"sure thing {w}")]}
        for i, w in [(1, "one"), (2, "two"), (3, "three")]]


def test_session_score_is_best_plus_second_best():
    p = {"date": "2023/06/05 (Mon) 10:00", "turns": [U("greyhound pixel sister"), A("nice"), U("greyhound pixel")]}
    q = {"date": "2023/06/06 (Tue) 10:00", "turns": [U("greyhound pixel sister adopted"), A("lovely")]}
    idx = bm25_index(FILL + [p, q])                                   # p is session 3, q session 4
    question = "greyhound pixel sister adopted"
    s = idx.bm25_scores(question, 1.2, 0.75)
    start = idx._session_start
    assert s[start[4]] > s[start[3]] > s[start[3] + 2] > 0            # q's turn ranks 1st, p's 2nd and 3rd
    assert list(idx.retrieve(question).chosen)[:2] == [3, 4]          # 1/62 + 1/63 beats 1/61
    assert list(idx.retrieve(question, config={"second_weight": 0.0}).chosen)[:2] == [4, 3]


def test_equal_turn_scores_break_ties_by_content_hash_not_position():
    a = {"date": "2023/06/05 (Mon) 10:00", "turns": [U("greyhound pixel beach"), A("fine one")]}
    b = {"date": "2023/06/06 (Tue) 10:00", "turns": [U("pixel greyhound beach"), A("fine two")]}
    first = min((a, b), key=lambda s: turn_key("user", s["turns"][0]["content"]))
    one = R.SESSION_HEADER_TOKENS + sum(words(json.dumps(t)) + 1 for t in a["turns"])
    for pair in ([a, b], [b, a]):
        idx = bm25_index(FILL + pair)
        s = idx.bm25_scores("greyhound pixel beach", 1.2, 0.75)
        assert s[idx._session_start[3]] == s[idx._session_start[4]] > 0   # an exact tie
        r = idx.retrieve("greyhound pixel beach", budget=one)             # room for one of the two
        assert [idx.sessions[si][1][0]["content"] for si in r.chosen] == [first["turns"][0]["content"]]


def test_equal_session_scores_break_ties_by_v10_session_hash():
    names = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    sessions = [{"date": f"2023/07/0{i + 1} (Sat) 10:00", "turns": [U(f"note {w}"), A(f"ok {w}")]}
                for i, w in enumerate(names)]
    v10_hash = [turn_key("session", json.dumps(s, sort_keys=True)) for s in sessions]
    r = bm25_index(sessions).retrieve("zebra xylophone")             # no match anywhere: all sessions score 0
    assert list(r.chosen) == sorted(range(len(sessions)), key=lambda i: v10_hash[i])


def test_dense_scores_are_cosines_with_user_turns():
    idx, enc = build()
    s = idx.dense_scores(enc.embed_query(QUESTION))
    pos = idx._session_start[3] + 4                                   # the Beatrice turn
    assert np.nanargmax(s) == pos
    want = float(np.dot(enc._vec(history()[3]["turns"][4]["content"]).astype(np.float16).astype(np.float32),
                        enc._vec(QUESTION).astype(np.float16).astype(np.float32)))
    assert abs(float(s[pos]) - want) < 1e-6


class FixedEncoder(StubEncoder):
    """Returns the vectors a test chose for each text."""

    def __init__(self, table):
        super().__init__()
        self.table = table

    def _vec(self, text):
        return np.asarray(self.table[text], dtype=np.float32)


def test_dense_ranks_negative_cosines_too():
    s = [{"date": "2023/09/01 (Fri) 10:00", "turns": [U("north"), A("x")]},
         {"date": "2023/09/02 (Sat) 10:00", "turns": [U("south"), A("y")]}]
    tie_winner = min((0, 1), key=lambda i: turn_key("session", json.dumps(s[i], sort_keys=True)))
    other = 1 - tie_winner
    e = lambda *x: np.pad(np.array(x, np.float32), (0, StubEncoder.DIM - len(x)))
    table = {s[tie_winner]["turns"][0]["content"]: e(0.0, 1.0), s[other]["turns"][0]["content"]: e(1.0, 0.0)}
    idx = MemoryIndex(config={"channels": ["dense"]}, encoder=FixedEncoder(table), token_counter=words)
    idx.add_sessions(s)
    q = e(-1.0, -3.0) / np.sqrt(10.0)                        # cosines: -0.32 (other), -0.95 (tie winner)
    one = R.SESSION_HEADER_TOKENS + sum(words(json.dumps(t)) + 1 for t in s[0]["turns"])
    r = idx.retrieve("anything", query_embedding=q, budget=one)
    assert list(r.chosen) == [other]                         # all cosines ranked, not only positive ones


def test_retrieve_passes_bm25_k1_and_b_in_order():
    idx = bm25_index(history())
    calls, real = [], idx.bm25_scores
    idx.bm25_scores = lambda q, k1, b: calls.append((k1, b)) or real(q, k1, b)
    idx.retrieve(QUESTION)
    idx.retrieve(QUESTION, config={"bm25_k1": 0.9, "bm25_b": 0.4})
    assert calls == [(1.2, 0.75), (0.9, 0.4)]


def test_repeated_query_terms_count_once():
    idx = bm25_index(history())
    assert np.array_equal(idx.bm25_scores("greyhound greyhound sister", 1.2, 0.75),
                          idx.bm25_scores("greyhound sister", 1.2, 0.75))


def test_only_the_top_whole_top_sessions_go_in_whole():
    long = [{"date": f"2023/08/0{d} (Tue) 10:00",
             "turns": [U(f"greyhound walk {d} {j}") if j % 2 == 0 else A(f"ok {d} {j}") for j in range(12)]}
            for d in range(1, 5)]
    idx = bm25_index(long)                                   # windows cover at most 10 of a session's 12 turns
    assert idx.retrieve("greyhound walk").whole == 4
    r = idx.retrieve("greyhound walk", config={"whole_top": 1})
    assert r.n_sessions == 4 and r.whole == 1 and r.chosen[list(r.chosen)[0]] is None


def test_w_assistant_scales_assistant_turns_only():
    a = {"date": "2023/09/03 (Sun) 10:00", "turns": [U("hello there"), A("the zeppelin landed")]}
    b = {"date": "2023/09/04 (Mon) 10:00", "turns": [U("I saw a zeppelin"), A("nice")]}
    idx = bm25_index(FILL + [a, b])                          # a is session 3, b session 4
    assert list(idx.retrieve("zeppelin", config={"w_assistant": 0.0}).chosen)[0] == 4
    assert list(idx.retrieve("zeppelin", config={"w_assistant": 10.0}).chosen)[0] == 3


def test_terms_are_stemmed_and_filtered():
    assert terms("I adopted 2 greyhounds, x-rays and running shoes") == ["adopt", "greyhound", "ray", "run", "shoe"]
    assert terms("The Greyhound RUNS; a b c 7 z") == ["greyhound", "run"]
    assert terms("Room 101 has 2 cats and 7 dogs") == ["room", "101", "cat", "dog"]   # digits are terms


def test_budget_accounting_is_v10s():
    assert (R.SESSION_HEADER_TOKENS, R.TURN_OVERHEAD_TOKENS) == (16, 1)
    turns = [U("café greyhound ☕"), A("naïve reply")]
    idx = MemoryIndex(config={"channels": ["bm25"]}, token_counter=len)   # characters: shows the serialisation
    idx.add_session(turns, date="2023/05/01 (Mon) 09:00")
    r = idx.retrieve("greyhound")
    assert r.tokens == 16 + sum(len(json.dumps(t, ensure_ascii=False)) + 1 for t in turns)
    assert r.text == render_history([("2023/05/01 (Mon) 09:00", turns)]) and "\\u00e9" in r.text


# ---------------------------------------------------------------- rendering

def test_render_is_the_official_builder_format():
    blocks = [("2023/05/01 (Mon) 09:00", [U("café ☕"), A("ok")])]
    assert render_history(blocks) == (
        '\n### Session 1:\nSession Date: 2023/05/01 (Mon) 09:00\nSession Content:\n\n'
        '[{"role": "user", "content": "caf\\u00e9 \\u2615"}, {"role": "assistant", "content": "ok"}]\n')
    assert "café ☕" in render_history(blocks, ensure_ascii=False)


# ---------------------------------------------------------------- exactness against v10.py's own formulas

def _v10_bm25(q_terms, docs, k1, b):          # v10.py's bm25(), verbatim
    n = len(docs)
    avgdl = sum(sum(d.values()) for d in docs) / max(n, 1) or 1.0
    df = collections.Counter(t for d in docs for t in d)
    qs = set(q_terms)
    out = np.zeros(n, dtype=np.float32)
    for i, d in enumerate(docs):
        dl = sum(d.values())
        s = 0.0
        for t in qs:
            f = d.get(t, 0)
            if f:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        out[i] = s
    return out


def _v10_rrf(scores, k, positive_only, tiebreak):   # v10.py's rrf(), verbatim
    valid = ~np.isnan(scores)
    filled = np.where(valid, scores, -np.inf)
    order = np.lexsort((tiebreak, -filled))
    r = np.zeros(len(scores), dtype=np.float32)
    rank = 0
    for i in order:
        if not valid[i] or (positive_only and scores[i] <= 0):
            break
        r[i] = 1.0 / (k + rank + 1)
        rank += 1
    return r


def test_bm25_equals_v10():
    idx, _ = build()
    docs = [collections.Counter(terms(t["content"])) for s in history() for t in s["turns"]]
    for q in [QUESTION, "spanish lessons practice", "laptop fans thermal games", "nothing matches zzz",
              "greyhound greyhound sister 101"]:
        assert np.array_equal(idx.bm25_scores(q, 1.2, 0.75), _v10_bm25(terms(q), docs, 1.2, 0.75))


def test_rrf_equals_v10():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 60))
        s = rng.choice([0.0, 0.5, 1.0, 2.0, -1.0], size=n).astype(np.float32) + (
            rng.random(n).astype(np.float32) * (rng.random() < 0.5))
        s[rng.random(n) < 0.3] = np.nan
        tb = rng.permutation(n)
        for pos in (True, False):
            assert np.array_equal(R.rrf(s, 60, pos, tb), _v10_rrf(s, 60, pos, tb))


def test_turn_key_is_v10s():
    assert turn_key("user", "hi") == hashlib.sha1(b"user\x00hi").hexdigest()


# ---------------------------------------------------------------- the bge encoder, without the model

def test_bge_prefixes_the_query_and_not_the_turns():
    from mnemosyne_v10 import encoders as E
    enc = E.BgeEncoder.__new__(E.BgeEncoder)                 # no torch, no download
    seen = []
    enc._embed = lambda texts: seen.append(list(texts)) or np.zeros((len(texts), E.DIM), np.float32)
    enc.embed_query("where is Pixel?")
    enc.embed_turns(["a turn", "another"])
    assert E.QUERY_PREFIX == "Represent this sentence for searching relevant passages: "
    assert seen == [[E.QUERY_PREFIX + "where is Pixel?"], ["a turn", "another"]]


def test_bge_pools_the_cls_token_and_keeps_input_order():
    torch = pytest.importorskip("torch")
    from mnemosyne_v10 import encoders as E
    enc = E.BgeEncoder.__new__(E.BgeEncoder)
    enc._torch, enc.device, enc.batch_size = torch, "cpu", 2
    calls = []

    class Batch(dict):
        def to(self, device):
            return self

    def tokenizer(texts, **kw):
        calls.append(kw)
        return Batch(lens=[len(t) for t in texts])

    def model(lens):                          # CLS row points along axis len(text); the other rows elsewhere
        h = torch.zeros(len(lens), 3, E.DIM)
        for i, n in enumerate(lens):
            h[i, 0, n] = 2.0
            h[i, 1:, 0] = 5.0
        return types.SimpleNamespace(last_hidden_state=h)

    enc.tokenizer, enc.model = tokenizer, model
    texts = ["ccc", "a", "bbbb", "dd", "eeeee"]
    want = np.zeros((len(texts), E.DIM), np.float32)
    want[np.arange(len(texts)), [len(t) for t in texts]] = 1.0
    assert np.array_equal(enc._embed(texts), want)
    assert all(kw["truncation"] and kw["max_length"] == 512 and kw["padding"] for kw in calls)


# ---------------------------------------------------------------- optional: real tokenizer, scikit-learn

def _o200k_or_skip():
    try:
        return R.O200kCounter()
    except Exception as e:  # tiktoken missing, or its encoding file cannot be fetched offline
        pytest.skip(f"o200k_base unavailable: {e}")


def test_o200k_counter_counts_special_literals_as_text():
    c = _o200k_or_skip()
    assert c("hello world") == 2
    assert c("<|endoftext|>") > 1


def test_stopwords_match_scikit_learn():
    sk = pytest.importorskip("sklearn.feature_extraction.text")
    from mnemosyne_v10.stopwords import ENGLISH_STOP_WORDS
    assert ENGLISH_STOP_WORDS == sk.ENGLISH_STOP_WORDS


def test_cli_index_and_retrieve_bm25(tmp_path):
    _o200k_or_skip()
    hist = tmp_path / "history.json"
    hist.write_text(json.dumps({"sessions": history()}))
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    run = lambda *a: subprocess.run([sys.executable, "-m", "mnemosyne_v10", *a], env=env, cwd=tmp_path,
                                    capture_output=True, text=True, check=True)
    run("index", "--history", str(hist), "--out", str(tmp_path / "idx"), "--set", 'channels=["bm25"]')
    out = run("retrieve", "--index", str(tmp_path / "idx"), "--question", QUESTION, "--budget", "200",
              "--set", 'channels=["bm25"]', "--format", "json").stdout
    doc = json.loads(out)
    assert doc["tokens"] <= 200 and 3 in [s["index"] for s in doc["sessions"]]
    assert "Beatrice" in doc["text"]
    idx = MemoryIndex(config={"channels": ["bm25"]})                  # the same retrieval in-process
    idx.add_sessions(history())
    r = idx.retrieve(QUESTION, budget=200)
    assert r.tokens < 200                                             # so a "tokens" field that echoed the budget would show
    assert (doc["tokens"], doc["budget"], doc["n_sessions"], doc["whole"], doc["text"]) == (
        r.tokens, 200, r.n_sessions, r.whole, r.text)
