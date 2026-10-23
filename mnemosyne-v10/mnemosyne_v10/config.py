"""The frozen v10 retrieval configuration, and helpers to load or override it.

The defaults are read from ``frozen_v10.json``, a byte-identical copy of the configuration that was frozen
before the held-out evaluation (sha256 below; config hash ``7bc75f509aa0``). Constants that v10.py kept in code
rather than in the file live in ``retrieval.py`` and ``encoders.py``.
"""
import hashlib
import json
from importlib import resources

FROZEN_V10_FILE = "frozen_v10.json"
FROZEN_V10_FILE_SHA256 = "37e8361f5fecfffca75a2c20f0b62d8d46d3de5ca61cf0ddb682c9033dd704c5"
FROZEN_V10_HASH = "7bc75f509aa0"

CHANNELS = ("bm25", "dense", "time")
_KEYS = {
    "channels": list,
    "rrf_k": (int, float),
    "w_bm25": (int, float),
    "w_dense": (int, float),
    "w_assistant": (int, float),
    "second_weight": (int, float),
    "budget": int,
    "whole_top": int,
    "window": int,
    "window_seeds": int,
    "bm25_k1": (int, float),
    "bm25_b": (int, float),
    "w_time": (int, float),
}


def frozen_file_bytes() -> bytes:
    return resources.files(__package__).joinpath(FROZEN_V10_FILE).read_bytes()


def frozen_config() -> dict:
    """A fresh copy of the frozen v10 configuration."""
    return json.loads(frozen_file_bytes())


def config_hash(cfg: dict) -> str:
    """The 12-hex-digit hash v10.py records for a configuration (``cfg_hash``)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def make_config(overrides=None, base=None) -> dict:
    """Frozen defaults (or ``base``) updated with ``overrides``, validated.

    Unknown keys and unknown channels raise ``ValueError``, so a typo cannot silently fall back to a default.
    """
    cfg = dict(frozen_config() if base is None else base)
    for key, value in dict(overrides or {}).items():
        if key not in _KEYS:
            raise ValueError(f"unknown v10 config key {key!r}; known keys: {sorted(_KEYS)}")
        cfg[key] = value
    for key, types in _KEYS.items():
        if key not in cfg:
            raise ValueError(f"config is missing {key!r}")
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], types):
            raise ValueError(f"config {key!r} has type {type(cfg[key]).__name__}")
    cfg["channels"] = list(cfg["channels"])
    bad = [c for c in cfg["channels"] if c not in CHANNELS]
    if bad or not cfg["channels"]:
        raise ValueError(f"channels must be a non-empty subset of {CHANNELS}, got {cfg['channels']}")
    if cfg["budget"] < 0 or cfg["window"] < 0 or cfg["window_seeds"] < 1 or cfg["whole_top"] < 0:
        raise ValueError("budget, window and whole_top must be >= 0 and window_seeds >= 1")
    return cfg


def load_config(path=None, overrides=None) -> dict:
    """Frozen defaults, updated from a JSON file (optional), then from ``overrides`` (optional)."""
    cfg = make_config()
    if path is not None:
        with open(path) as f:
            cfg = make_config(json.load(f), base=cfg)
    return make_config(overrides, base=cfg)
