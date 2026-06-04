"""CLI entry point for ``python -m ara.agent``."""

from __future__ import annotations

import sys

from ara.agent.server import main

if __name__ == "__main__":
    sys.exit(main())
