"""Kintsugi CLI plugin commands — re-exports for patching compatibility."""

from kintsugi.cli.plugins import *  # noqa: F401,F403


class PluginManager:
    """Plugin manager used by CLI plugin commands."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def list_available(self):
        return []

    def list_installed(self):
        return []
