"""Mnemosyne v10: whole-session retrieval for agent memory, with the frozen configuration of the
LongMemEval_S-cleaned evaluation as its defaults.

    from mnemosyne_v10 import MemoryIndex, BgeEncoder

    index = MemoryIndex(encoder=BgeEncoder())
    index.add_session([{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
                      date="2026/09/25 (Fri) 21:00")
    r = index.retrieve("What did I say about ...?")
    context = r.text          # chosen sessions, chronological, <= ~24k o200k tokens
"""
from .config import (FROZEN_V10_FILE_SHA256, FROZEN_V10_HASH, config_hash, frozen_config, load_config,
                     make_config)
from .encoders import BgeEncoder, EmbeddingStore
from .retrieval import (MemoryIndex, O200kCounter, Retrieval, SelectedSession, load_history, render_history,
                        terms, turn_key)

__version__ = "10.0.0"
__all__ = ["MemoryIndex", "Retrieval", "SelectedSession", "BgeEncoder", "EmbeddingStore", "O200kCounter",
           "frozen_config", "make_config", "load_config", "config_hash", "load_history", "render_history",
           "terms", "turn_key", "FROZEN_V10_HASH", "FROZEN_V10_FILE_SHA256", "__version__"]
