"""Coloured logger factory used throughout the Ara engine."""

from __future__ import annotations

import logging
from typing import TypeAlias

from ara.utils.ansi import BOLD, END, GREEN, LIGHTGREY, RED, YELLOW

_Level: TypeAlias = str | int


class ColorFormatter(logging.Formatter):
    """Formatter that applies ANSI colours per log level."""

    _format_brief = "%(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    _format_verbose = (
        "%(asctime)s - %(name)s: %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: LIGHTGREY + _format_verbose + END,
        logging.INFO: GREEN + _format_brief + END,
        logging.WARNING: YELLOW + _format_verbose + END,
        logging.ERROR: RED + _format_verbose + END,
        logging.CRITICAL: BOLD + RED + _format_verbose + END,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(name: str, level: _Level | None = None) -> logging.Logger:
    """Create or retrieve a logger with a coloured stream handler.

    :param name: Logger name ( conventionally ``__name__``).
    :param level: Optional override.  When ``None``, the level defaults to
        :data:`logging.DEBUG` when Python is run without ``-O``, otherwise
        :data:`logging.INFO`.
    :return: Configured logger instance.
    """
    ch = logging.StreamHandler()
    ch.setFormatter(ColorFormatter())

    logger = logging.getLogger(name)
    # Avoid adding duplicate handlers when a module is imported multiple times
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(ch)

    if level is None:
        logger.setLevel(logging.DEBUG if __debug__ else logging.INFO)
    else:
        logger.setLevel(level)

    return logger
