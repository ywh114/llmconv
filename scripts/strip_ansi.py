#!/usr/bin/env python3
"""Strip ANSI escape codes from stdin and write clean text to stdout."""
import re
import sys

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def strip(text: str) -> str:
    return ANSI_RE.sub("", text)

if __name__ == "__main__":
    sys.stdout.write(strip(sys.stdin.read()))
