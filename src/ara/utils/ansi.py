"""ANSI escape sequences for terminal color formatting."""

from __future__ import annotations

BLACK = "\033[30m"
DARKGRAY = DARKGREY = "\033[90m"

DARKRED = "\033[31m"
RED = "\033[91m"

DARKGREEN = "\033[32m"
GREEN = "\033[92m"

DARKYELLOW = "\033[33m"
YELLOW = "\033[93m"

DARKBLUE = "\033[34m"
BLUE = "\033[94m"

DARKMAGENTA = "\033[35m"
MAGENTA = "\033[95m"

DARKCYAN = "\033[36m"
CYAN = "\033[96m"

LIGHTGRAY = LIGHTGREY = "\033[37m"
WHITE = "\033[97m"

UNDERLINE = "\033[4m"
NOUNDERLINE = "\033[24m"
BOLD = "\033[1m"
NOBOLD = "\033[21m"

RESET = "\033[39m\033[49m"
CLEAR = "\033[2K"

END = "\033[0m"
