"""Web-based VN frontend for Ara."""

from __future__ import annotations

__all__ = ["AgentProxy", "create_app"]

from ara.webclient.proxy import AgentProxy
from ara.webclient.server import create_app
