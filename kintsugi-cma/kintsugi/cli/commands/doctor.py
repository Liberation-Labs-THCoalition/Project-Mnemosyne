"""Kintsugi CLI doctor commands — re-exports for patching compatibility."""

from kintsugi.cli.doctor import *  # noqa: F401,F403


def get_db_connection(**kwargs):
    """Get database connection for doctor diagnostics."""
    return None


class APIClient:
    """API client used by doctor diagnostics."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    async def health_check(self):
        return {"status": "ok"}
