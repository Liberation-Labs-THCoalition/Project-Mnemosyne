"""Mnemosyne v10 retrieval: index a conversation history, retrieve whole sessions under a token budget.

A port of ``retrieve()`` from v10.py (the benchmark harness as run) into an index you build once and query
per turn. Scoring and packing follow v10.py step for step, so the frozen configuration reproduces the
published held-out retrieval record (see ``tests/test_parity.py``):

1. **BM25** over every turn, user and assistant (k1 = 1.2, b = 0.75; Porter-stemmed ASCII terms, stop words
   and 1-character terms dropped). The corpus is the indexed history.
2. **Dense**: cosine between the question and each **user** turn (see ``encoders.py``).
3. **Weighted reciprocal-rank fusion** per turn: weight / (k + rank), k = 60, BM25 1.0, dense 2.0. Ties are
   broken by the rank of the turn's content hash, never by its position.
4. **Session score** = best fused turn + ``second_weight`` x second-best (1.0: best + second-best).
5. **Packing**: sessions in score order; each goes in whole if it fits the remaining budget (24,000 o200k
   tokens), otherwise a window of +/-2 turns around its two best turns goes in, if that fits.
6. **Rendering**: the chosen sessions (or windows) in chronological order, in the history format of the
   official LongMemEval prompt builder that the benchmark reader saw.

The retriever sees only (role, content) per turn and a date per session. No LLM calls.
"""
import collections
import datetime as dt
import functools
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass

import numpy as np
from nltk.stem import PorterStemmer

from . import encoders as _enc
from .config import make_config
from .stopwords import ENGLISH_STOP_WORDS

# Budget accounting, as in v10.py: each turn costs its o200k tokens as json.dumps({"role", "content"},
# ensure_ascii=False) plus 1, and each session a fixed header allowance of 16 ("### Session N:\nSession Date:
# ...\nSession Content:\n" in the official builder).
SESSION_HEADER_TOKENS = 16
TURN_OVERHEAD_TOKENS = 1
HISTORY_FORMAT = "mnemosyne-v10/history"


def turn_key(role, content):
    """sha1(role NUL content): the key v10 uses for caches, tie-breaks and stored embeddings."""
    return hashlib.sha1((role + "\x00" + content).encode()).hexdigest()


# ---------------------------------------------------------------- terms

_stem = functools.lru_cache(maxsize=1 << 20)(PorterStemmer().stem)
_TOKEN = re.compile(r"[a-z0-9]+")


def terms(text):
    """BM25 terms: lower-cased ASCII alphanumeric runs, minus stop words and 1-character runs, Porter-stemmed."""
    return [_stem(w) for w in _TOKEN.findall(text.lower()) if len(w) > 1 and w not in ENGLISH_STOP_WORDS]


@functools.lru_cache(maxsize=1 << 18)
def _term_counts(content):
    return collections.Counter(terms(content))   # shared between indexes: treat as read-only


# ---------------------------------------------------------------- token counting

class O200kCounter:
    """Counts tokens with tiktoken's o200k_base, as v10's budget does.

    Special-token literals (e.g. a turn containing the text ``<|endoftext|>``) count as ordinary text.
    """

    def __init__(self):
        import tiktoken
        self._enc = tiktoken.get_encoding("o200k_base")
        self.count = functools.lru_cache(maxsize=1 << 18)(self._count)

    def _count(self, text):
        return len(self._enc.encode(text, disallowed_special=()))

    def __call__(self, text):
        return self.count(text)


_DEFAULT_COUNTER = None


def default_token_counter():
    global _DEFAULT_COUNTER
    if _DEFAULT_COUNTER is None:
        _DEFAULT_COUNTER = O200kCounter()
    return _DEFAULT_COUNTER


# ---------------------------------------------------------------- the time channel (off in the frozen config)
# Relative-time phrases in the question become date windows around the question date (v10.py, unchanged).
# Session and question dates must start with YYYY/MM/DD for this channel.

_NUMW = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2, "couple of": 2,
         "a few": 3, "few": 3, "several": 4}
_NUMRE = r"(\d+|a couple of|couple of|a few|few|several|an|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_UNIT = {"day": 1.0, "week": 7.0, "month": 30.44, "year": 365.25}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december"]


def parse_date(s):
    return dt.datetime.strptime(s[:10], "%Y/%m/%d")


def time_windows(question, question_date):
    q, qd, D = question.lower(), parse_date(question_date), lambda x: dt.timedelta(days=x)
    num = lambda w: int(w) if w.isdigit() else _NUMW[w]
    wins = []
    for m in re.finditer(_NUMRE + r"\s+(day|week|month|year)s?\s+ago\b", q):
        u = _UNIT[m.group(2)]; c = qd - D(num(m.group(1)) * u); tol = max(1.0, 0.5 * u)
        wins.append((c - D(tol), c + D(tol)))
    for m in re.finditer(r"\b(?:past|last)\s+" + _NUMRE + r"\s+(day|week|month|year)s\b", q):
        wins.append((qd - D(num(m.group(1)) * _UNIT[m.group(2)] + 1), qd))
    for m in re.finditer(r"\blast (week|weekend|month|year)\b", q):
        span = {"week": 14, "weekend": 9, "month": 62, "year": 730}[m.group(1)]
        wins.append((qd - D(span), qd))
    if re.search(r"\byesterday\b", q):
        wins.append((qd - D(2), qd))
    for m in re.finditer(r"\b(?:last|this past) (" + "|".join(_WEEKDAYS) + r")\b", q):
        back = (qd.weekday() - _WEEKDAYS.index(m.group(1))) % 7 or 7
        d = qd - D(back); wins.append((d - D(1), d + D(1)))
    for m in re.finditer(r"\bthis (week|month)\b", q):
        wins.append((qd - D(7 if m.group(1) == "week" else 31), qd))
    for m in re.finditer(r"\b(?:in|during|since) (" + "|".join(_MONTHS) + r")\b", q):
        mi = _MONTHS.index(m.group(1)) + 1
        y = qd.year if mi <= qd.month else qd.year - 1
        start = dt.datetime(y, mi, 1)
        end = dt.datetime(y + (mi == 12), mi % 12 + 1, 1) - D(1)
        wins.append((start, qd if "since" in m.group(0) else end))
    return wins


# ---------------------------------------------------------------- fusion

def rrf(scores, k, positive_only, tiebreak):
    """Reciprocal-rank weights 1 / (k + rank), rank from 1. NaN scores (no signal) get no rank and no weight;
    with ``positive_only``, neither do scores <= 0. Equal scores are ordered by ``tiebreak``."""
    valid = ~np.isnan(scores)
    filled = np.where(valid, scores, -np.inf)
    order = np.lexsort((tiebreak, -filled))
    m = int(np.count_nonzero(valid & (filled > 0))) if positive_only else int(np.count_nonzero(valid))
    r = np.zeros(len(scores), dtype=np.float32)
    r[order[:m]] = 1.0 / (k + np.arange(m) + 1)
    return r


# ---------------------------------------------------------------- rendering

def render_history(blocks, ensure_ascii=True):
    """Render (date, turns) blocks, already in the order to show, in the official LongMemEval builder's
    history format ('orig-session', JSON). ``ensure_ascii=True`` matches the builder byte for byte;
    ``False`` keeps non-ASCII text unescaped (fewer tokens, not what the benchmark reader saw)."""
    return "".join(
        "\n### Session {}:\nSession Date: {}\nSession Content:\n\n{}\n".format(
            i, date, json.dumps(turns, ensure_ascii=ensure_ascii))
        for i, (date, turns) in enumerate(blocks, 1))


@dataclass(frozen=True)
class SelectedSession:
    index: int          # position of the session in the indexed history
    date: str
    turns: tuple        # the rendered turns, as ({"role", "content"}, ...)
    turn_indexes: tuple  # indexes of those turns within the session
    whole: bool


@dataclass
class Retrieval:
    """What ``MemoryIndex.retrieve`` selected. ``sessions`` is in chronological (rendering) order."""
    question: str
    sessions: list
    tokens: int           # budget tokens used, v10 accounting
    budget: int
    chosen: dict          # {session index: None (whole) | [turn indexes]}, in selection (score) order
    time_windows: int = 0

    @property
    def n_sessions(self):
        return len(self.sessions)

    @property
    def whole(self):
        return sum(s.whole for s in self.sessions)

    def render(self, ensure_ascii=True):
        return render_history(((s.date, list(s.turns)) for s in self.sessions), ensure_ascii=ensure_ascii)

    @property
    def text(self):
        return self.render()


# ---------------------------------------------------------------- the index

def _clean_turns(turns):
    out = []
    for t in turns:
        role, content = t["role"], t["content"]
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("turn role and content must be strings")
        out.append({"role": role, "content": content})   # only (role, content) ever reaches retrieval
    return out


class MemoryIndex:
    """A conversation history (sessions of turns) indexed for v10 retrieval.

    ``config``        overrides of the frozen v10 configuration (default: none, i.e. frozen v10).
    ``encoder``       embeds user turns and questions for the dense channel (e.g. ``BgeEncoder()``).
    ``embeddings``    an ``EmbeddingStore`` of precomputed turn vectors; turns missing from it are embedded
                      with ``encoder`` and added to it.
    ``token_counter`` text -> token count for the budget (default: tiktoken o200k_base, as v10).

    The dense channel needs an encoder or a store at indexing time, and at query time an encoder or an
    explicit ``query_embedding``. A BM25-only index: ``MemoryIndex(config={"channels": ["bm25"]})``.
    """

    def __init__(self, config=None, encoder=None, embeddings=None, token_counter=None):
        self.config = make_config(config)
        self.encoder = encoder
        self.embeddings = embeddings
        if self.embeddings is None and encoder is not None:
            self.embeddings = _enc.EmbeddingStore(meta=getattr(encoder, "meta", None))
        if encoder is not None and self.embeddings is not None and self.embeddings.meta and getattr(encoder, "meta", None):
            if _enc.space_of(encoder.meta) != _enc.space_of(self.embeddings.meta):
                raise ValueError("the encoder and the embedding store describe different embedding spaces: "
                                 f"{_enc.space_of(encoder.meta)} vs {_enc.space_of(self.embeddings.meta)}")
        self._count = token_counter or default_token_counter()
        self.sessions = []        # [(date, [{"role", "content"}, ...])]
        self._keys = []           # per turn, flat order (session, then turn)
        self._flat = []           # per turn: (session index, turn index)
        self._assistant = []      # per turn: role == "assistant"
        self._cost = []           # per turn: budget tokens
        self._dl = []             # per turn: number of BM25 terms
        self._postings = collections.defaultdict(list)   # term -> [(turn, tf)]
        self._total_terms = 0
        self._user_pos = []       # flat positions of user turns
        self._user_rows = []      # their rows in self.embeddings
        self._session_hash = []   # per session: tie-break hash
        self._session_start = []  # per session: first flat position
        self._tb = None           # cached tie-break ranks
        self._dense = None        # cached float32 matrix of user-turn vectors

    # -------------------------------------------------------- building

    def __len__(self):
        return len(self.sessions)

    @property
    def n_turns(self):
        return len(self._keys)

    def _uses_dense(self, config=None):
        return "dense" in (config or self.config)["channels"]

    def add_session(self, turns, date):
        """Add one session: ``turns`` is a list of {"role", "content"} (other keys are ignored), ``date`` a
        string that sorts chronologically (LongMemEval uses "YYYY/MM/DD (Day) HH:MM"). Returns its index."""
        return self.add_sessions([{"date": date, "turns": turns}])[0]

    def add_sessions(self, sessions):
        """Add sessions given as {"date", "turns"} mappings or (date, turns) pairs. Returns their indexes."""
        prepared = []
        for s in sessions:
            date, turns = (s["date"], s["turns"]) if isinstance(s, dict) else s
            if not isinstance(date, str):
                raise TypeError("session date must be a string")
            prepared.append((date, _clean_turns(turns)))
        if self._uses_dense():
            self._embed_missing(t["content"] for _, turns in prepared for t in turns if t["role"] == "user")
        first = len(self.sessions)
        for date, turns in prepared:
            self._append(date, turns)
        self._tb = self._dense = None
        return list(range(first, len(self.sessions)))

    def _embed_missing(self, user_texts):
        if self.embeddings is None:
            raise ValueError("the dense channel needs an encoder or an EmbeddingStore "
                             "(or use config={'channels': ['bm25']})")
        todo = {}
        for text in user_texts:
            k = turn_key("user", text)
            if k not in self.embeddings and k not in todo:
                todo[k] = text
        if not todo:
            return
        if self.encoder is None:
            raise KeyError(f"{len(todo)} user turn(s) have no stored embedding and no encoder was given")
        keys = list(todo)
        self.embeddings.add(keys, self.encoder.embed_turns([todo[k] for k in keys]))

    def _append(self, date, turns):
        si = len(self.sessions)
        self.sessions.append((date, turns))
        self._session_start.append(len(self._keys))
        self._session_hash.append(turn_key("session", json.dumps({"date": date, "turns": turns}, sort_keys=True)))
        dense = self._uses_dense()
        for ti, t in enumerate(turns):
            pos = len(self._keys)
            k = turn_key(t["role"], t["content"])
            self._keys.append(k)
            self._flat.append((si, ti))
            self._assistant.append(t["role"] == "assistant")
            self._cost.append(self._count(json.dumps(t, ensure_ascii=False)) + TURN_OVERHEAD_TOKENS)
            tf = _term_counts(t["content"])
            dl = sum(tf.values())
            self._dl.append(dl)
            self._total_terms += dl
            for term, f in tf.items():
                self._postings[term].append((pos, f))
            if t["role"] == "user" and dense:
                self._user_pos.append(pos)
                self._user_rows.append(self.embeddings.rows([k])[0])

    # -------------------------------------------------------- scoring

    def _tiebreak(self):
        if self._tb is None:
            self._tb = np.argsort(np.argsort(np.array(self._keys)))   # rank of each turn's content hash
        return self._tb

    def _dense_matrix(self):
        if self._dense is None:
            self._dense = np.asarray(self.embeddings.vectors(self._user_rows), dtype=np.float32)
        return self._dense

    def bm25_scores(self, question, k1, b):
        n = len(self._keys)
        avgdl = self._total_terms / max(n, 1) or 1.0
        s = np.zeros(n, dtype=np.float64)
        for t in sorted(set(terms(question))):
            post = self._postings.get(t)
            if not post:
                continue
            df = len(post)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, f in post:
                s[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self._dl[i] / avgdl))
        return s.astype(np.float32)

    def dense_scores(self, query_vector):
        out = np.full(len(self._keys), np.nan, dtype=np.float32)
        if self._user_pos:
            q = np.asarray(_enc.to_f16(query_vector), dtype=np.float32)   # stored as float16, as in v10
            out[self._user_pos] = self._dense_matrix() @ q
        return out

    # -------------------------------------------------------- retrieval

    def retrieve(self, question, question_date=None, query_embedding=None, budget=None, config=None):
        """Select sessions for ``question`` under the token budget.

        ``question_date`` is needed only by the time channel (off in the frozen config).
        ``query_embedding`` skips the encoder (e.g. a precomputed question vector).
        ``budget`` / ``config`` override the index's configuration for this call only.
        """
        cfg = self.config if config is None else make_config(config, base=self.config)
        if budget is not None:
            cfg = make_config({"budget": int(budget)}, base=cfg)
        n = len(self._keys)
        tb = self._tiebreak() if n else np.zeros(0, dtype=np.int64)
        fused = np.zeros(n, dtype=np.float32)
        if "bm25" in cfg["channels"]:
            fused += cfg["w_bm25"] * rrf(self.bm25_scores(question, cfg["bm25_k1"], cfg["bm25_b"]),
                                         cfg["rrf_k"], positive_only=True, tiebreak=tb)
        if "dense" in cfg["channels"]:
            if not self._uses_dense():
                raise ValueError("this index was built without the dense channel")
            if query_embedding is None:
                if self.encoder is None:
                    raise ValueError("the dense channel needs an encoder or a query_embedding")
                query_embedding = self.encoder.embed_query(question)
            fused += cfg["w_dense"] * rrf(self.dense_scores(query_embedding), cfg["rrf_k"],
                                          positive_only=False, tiebreak=tb)
        asst = np.array(self._assistant, dtype=bool)
        fused = np.where(asst, fused * cfg["w_assistant"], fused)

        per_sess = collections.defaultdict(list)
        for (si, ti), f in zip(self._flat, fused):
            per_sess[si].append((f, ti))
        sess_score = {}
        for si, lst in per_sess.items():
            top = sorted(lst, reverse=True)
            sess_score[si] = top[0][0] + cfg["second_weight"] * (top[1][0] if len(top) > 1 else 0.0)
        wins = []
        if "time" in cfg["channels"]:
            if question_date is None:
                raise ValueError("the time channel needs question_date")
            wins = time_windows(question, question_date)
        if wins:
            dist = {}
            for si in sess_score:
                d = parse_date(self.sessions[si][0])
                ds = [abs((d - (a + (b - a) / 2)).days) for a, b in wins if a <= d <= b]
                if ds:
                    dist[si] = min(ds)
            for rank, si in enumerate(sorted(dist, key=lambda si: (dist[si], self._session_hash[si]))):
                sess_score[si] += cfg["w_time"] / (cfg["rrf_k"] + rank + 1)
        order = sorted(sess_score, key=lambda si: (-sess_score[si], self._session_hash[si]))

        def cost(si, tis):
            start = self._session_start[si]
            return SESSION_HEADER_TOKENS + sum(self._cost[start + ti] for ti in tis)

        chosen, used = {}, 0
        for rank, si in enumerate(order):
            n_turns = len(self.sessions[si][1])
            if rank < cfg["whole_top"]:
                c = cost(si, range(n_turns))
                if used + c <= cfg["budget"]:
                    chosen[si], used = None, used + c
                    continue
            seeds = [ti for _, ti in sorted(per_sess[si], reverse=True)[:cfg["window_seeds"]]]
            tis = sorted({j for ti in seeds
                          for j in range(max(0, ti - cfg["window"]), min(n_turns, ti + cfg["window"] + 1))})
            c = cost(si, tis)
            if used + c <= cfg["budget"]:
                chosen[si], used = (None if len(tis) == n_turns else tis), used + c

        selected = []
        for si in sorted(chosen, key=lambda si: (self.sessions[si][0], si)):
            date, turns = self.sessions[si]
            tis = tuple(range(len(turns))) if chosen[si] is None else tuple(chosen[si])
            selected.append(SelectedSession(index=si, date=date, turns=tuple(turns[ti] for ti in tis),
                                            turn_indexes=tis, whole=chosen[si] is None))
        return Retrieval(question=question, sessions=selected, tokens=used, budget=cfg["budget"],
                         chosen=chosen, time_windows=len(wins))

    # -------------------------------------------------------- persistence

    def save(self, directory):
        """Write ``history.json`` and, for a dense index, the user-turn vectors under ``emb/``."""
        os.makedirs(directory, exist_ok=True)
        doc = {"format": HISTORY_FORMAT, "version": 1,
               "sessions": [{"date": d, "turns": turns} for d, turns in self.sessions]}
        _enc._atomic_write(os.path.join(directory, "history.json"),
                           lambda f: json.dump(doc, f, ensure_ascii=False))
        if self._uses_dense() and self.embeddings is not None:
            keys = list(dict.fromkeys(self._keys[p] for p in self._user_pos))
            self.embeddings.save(os.path.join(directory, "emb"), keys=keys)

    @classmethod
    def load(cls, directory, config=None, encoder=None, token_counter=None):
        """Re-open an index written by ``save()``."""
        with open(os.path.join(directory, "history.json"), encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("format") != HISTORY_FORMAT:
            raise ValueError(f"{directory}/history.json is not a {HISTORY_FORMAT} file")
        emb_dir = os.path.join(directory, "emb")
        store = _enc.EmbeddingStore.open(emb_dir) if os.path.exists(os.path.join(emb_dir, "turn_index.json")) else None
        index = cls(config=config, encoder=encoder, embeddings=store, token_counter=token_counter)
        index.add_sessions(doc["sessions"])
        return index


def load_history(path):
    """Sessions from a JSON file: {"sessions": [{"date", "turns"}]} (as ``save()`` writes) or a bare list."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    sessions = doc["sessions"] if isinstance(doc, dict) else doc
    return [{"date": s["date"], "turns": _clean_turns(s["turns"])} for s in sessions]
