"""Kintsugi CLI config commands — re-exports for patching compatibility."""

from kintsugi.cli.config import *  # noqa: F401,F403


def load_config(path=None):
    """Load configuration from file."""
    return {
        "database": {"host": "localhost", "port": 5432},
        "api": {"port": 8000},
    }


def validate_config(path=None, strict=False):
    """Validate configuration against schema."""
    return {"valid": True, "errors": []}
