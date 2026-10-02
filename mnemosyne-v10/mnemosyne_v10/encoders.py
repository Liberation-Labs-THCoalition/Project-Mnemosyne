"""Dense-channel embeddings for v10: the pinned bge encoder, and a store of precomputed turn vectors.

v10's dense channel embeds **user turns only** with BAAI/bge-base-en-v1.5 at a pinned revision: CLS pooling,
L2-normalised, inputs truncated to 512 tokens, and bge's retrieval prefix on the question only. Vectors are
stored as float16 and scored in float32, as in v10.

``EmbeddingStore`` reads and writes the on-disk layout of v10's ``embed_turns.py`` (``turn_index.json``,
``turns.f16.npy`` and ``meta.json``), so the published run's precomputed embeddings open unchanged and an
agent's index persists its vectors in the same format.
"""
import json
import os
import sys

import numpy as np

MODEL = "BAAI/bge-base-en-v1.5"
REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
POOLING = "cls+l2"
MAX_LENGTH = 512
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
ROLES = ("user",)
DIM = 768

# The fields that define the embedding space. Two stores (or a store and an encoder) whose values differ
# here must not be mixed.
SPACE_FIELDS = ("model", "revision", "pooling", "max_length", "query_prefix")
V10_META = {"model": MODEL, "revision": REVISION, "pooling": POOLING, "max_length": MAX_LENGTH,
            "roles": list(ROLES), "query_prefix": QUERY_PREFIX}


def space_of(meta):
    return {k: (meta or {}).get(k) for k in SPACE_FIELDS}


def to_f16(vectors):
    return np.asarray(vectors, dtype=np.float32).astype(np.float16)


class EmbeddingStore:
    """turn key -> float16 vector.

    ``open()`` memory-maps an existing directory read-only; ``add()`` keeps new rows in memory; ``save()``
    writes a (possibly different) directory. Nothing is ever written back into an opened directory unless
    ``save()`` is pointed at it.
    """

    def __init__(self, meta=None, dim=None):
        self.meta = dict(meta) if meta else None
        self.dim = dim
        self._keys = []
        self._row = {}
        self._base = None          # memory-mapped rows from open()
        self._new = []             # float16 arrays appended by add()
        self._n_base = 0

    @classmethod
    def open(cls, directory):
        with open(os.path.join(directory, "turn_index.json"), encoding="utf-8") as f:
            keys = json.load(f)
        meta = None
        meta_p = os.path.join(directory, "meta.json")
        if os.path.exists(meta_p):
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
        base = np.load(os.path.join(directory, "turns.f16.npy"), mmap_mode="r")
        if base.ndim != 2 or base.shape[0] < len(keys) or base.dtype != np.float16:
            raise ValueError(f"{directory}: turns.f16.npy has shape {base.shape} / dtype {base.dtype}, "
                             f"expected ({len(keys)}, d) float16")
        store = cls(meta=meta, dim=base.shape[1] if len(keys) else None)
        store._keys = list(keys)
        store._row = {k: i for i, k in enumerate(store._keys)}
        store._base = base
        store._n_base = len(keys)
        return store

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._row

    def rows(self, keys):
        missing = [k for k in keys if k not in self._row]
        if missing:
            raise KeyError(f"{len(missing)} turn(s) have no stored embedding (first key {missing[0]})")
        return [self._row[k] for k in keys]

    def vectors(self, rows):
        """float16 array of the given rows, in the given order."""
        rows = list(rows)
        out = np.empty((len(rows), self.dim or 0), dtype=np.float16)
        if not rows:
            return out
        idx = np.asarray(rows)
        in_base = idx < self._n_base
        if in_base.any():
            out[in_base] = self._base[idx[in_base]]
        if (~in_base).any():
            new = np.concatenate(self._new) if len(self._new) > 1 else self._new[0]
            self._new = [new]
            out[~in_base] = new[idx[~in_base] - self._n_base]
        return out

    def add(self, keys, vectors):
        """Add vectors for keys not already stored. ``vectors`` is (len(keys), dim); stored as float16."""
        vectors = to_f16(vectors)
        if vectors.ndim != 2 or vectors.shape[0] != len(keys):
            raise ValueError(f"expected {len(keys)} vectors, got shape {vectors.shape}")
        if self.dim is None:
            self.dim = vectors.shape[1]
        elif vectors.shape[1] != self.dim:
            raise ValueError(f"vector dim {vectors.shape[1]} != store dim {self.dim}")
        keep = []
        for i, k in enumerate(keys):
            if k not in self._row:
                self._row[k] = len(self._keys)
                self._keys.append(k)
                keep.append(i)
        if keep:
            self._new.append(vectors[keep])

    def save(self, directory, keys=None):
        """Write ``turn_index.json``, ``turns.f16.npy`` and ``meta.json`` (atomically, file by file).

        ``keys`` limits the saved rows to those keys (in that order); by default every stored row is saved.
        """
        os.makedirs(directory, exist_ok=True)
        keys = list(self._keys if keys is None else keys)
        rows = self.rows(keys)
        vecs = self.vectors(rows)
        meta = dict(self.meta or {})
        meta["turns"] = len(keys)
        _atomic_write(os.path.join(directory, "turns.f16.npy"), lambda f: np.save(f, vecs), binary=True)
        _atomic_write(os.path.join(directory, "turn_index.json"), lambda f: json.dump(keys, f))
        _atomic_write(os.path.join(directory, "meta.json"), lambda f: json.dump(meta, f, indent=1))


def _atomic_write(path, write, binary=False):
    tmp = f"{path}.tmp{os.getpid()}"
    with (open(tmp, "wb") if binary else open(tmp, "w", encoding="utf-8")) as f:
        write(f)
    os.replace(tmp, path)


class BgeEncoder:
    """BAAI/bge-base-en-v1.5 at v10's pinned revision (needs ``torch`` and ``transformers``).

    Runs on CPU unless you pass ``device``: on a shared machine, choose the GPU deliberately.
    """

    meta = dict(V10_META)

    def __init__(self, device="cpu", batch_size=16, threads=None, local_files_only=False):
        if os.environ.get("MNEMOSYNE_V10_HIDE_TORCHVISION") == "1":
            # Opt-in: a torchvision built for another torch ("operator torchvision::nms does not exist") breaks
            # transformers' BERT import. bge does not use torchvision, so hide it from this process.
            sys.modules.setdefault("torchvision", None)
        import torch
        from transformers import AutoModel, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(MODEL, revision=REVISION, local_files_only=local_files_only)
        self.model.eval().to(device)

    def _embed(self, texts):
        torch = self._torch
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            enc = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                 max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                cls = self.model(**enc).last_hidden_state[:, 0]
            out[idx] = torch.nn.functional.normalize(cls, dim=-1).float().cpu().numpy()
        return out

    def embed_turns(self, texts):
        """(n, 768) float32, L2-normalised. Turns get no prefix."""
        return self._embed(list(texts))

    def embed_query(self, text):
        """(768,) float32, L2-normalised, with bge's retrieval prefix."""
        return self._embed([QUERY_PREFIX + text])[0]
