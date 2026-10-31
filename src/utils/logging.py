"""Project-wide logging.

Rule for this codebase: no bare ``print`` in ``src/``. Every module gets a
logger via :func:`get_logger`; :func:`setup_logging` is called once by a script
or app entrypoint. Long-running stages should be wrapped in :class:`log_stage`
so timings land in the log rather than in someone's memory.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

__all__ = ["setup_logging", "get_logger", "log_stage"]

_CONFIGURED = False

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATEFMT = "%H:%M:%S"


class _ColourFormatter(logging.Formatter):
    """Adds ANSI colour when stderr is a TTY. Plain text otherwise."""

    _COLOURS = {
        logging.DEBUG: "\033[38;5;244m",
        logging.INFO: "\033[38;5;39m",
        logging.WARNING: "\033[38;5;214m",
        logging.ERROR: "\033[38;5;196m",
        logging.CRITICAL: "\033[1;38;5;196m",
    }
    _RESET = "\033[0m"

    def __init__(self, *, colour: bool) -> None:
        super().__init__(fmt=_FMT, datefmt=_DATEFMT)
        self.colour = colour

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self.colour:
            return text
        return f"{self._COLOURS.get(record.levelno, '')}{text}{self._RESET}"


def setup_logging(
    level: int | str = logging.INFO,
    *,
    log_file: str | Path | None = None,
    force: bool = False,
) -> None:
    """Configure the root logger. Safe to call more than once.

    Args:
        level: Threshold for the console handler.
        log_file: If given, everything at DEBUG and above is also written here
            (uncoloured), so a run leaves a complete audit trail.
        force: Reconfigure even if already set up.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(level)
    console.setFormatter(_ColourFormatter(colour=sys.stderr.isatty()))
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(fmt=_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
        root.addHandler(file_handler)

    # These libraries are chatty at INFO and drown out our own output.
    for noisy in ("urllib3", "matplotlib", "sentence_transformers", "transformers",
                  "huggingface_hub", "filelock", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Use ``get_logger(__name__)``."""
    return logging.getLogger(name)


@contextmanager
def log_stage(logger: logging.Logger, name: str, level: int = logging.INFO) -> Iterator[dict]:
    """Time a pipeline stage and log start/finish.

    Yields a dict the caller can populate; its contents are appended to the
    completion line, e.g. ``{"rows": 12345}`` -> ``... [rows=12345]``.
    """
    logger.log(level, "%s ...", name)
    started = time.perf_counter()
    payload: dict = {}
    try:
        yield payload
    except Exception:
        logger.error("%s FAILED after %.2fs", name, time.perf_counter() - started)
        raise
    extra = " ".join(f"{k}={v}" for k, v in payload.items())
    suffix = f" [{extra}]" if extra else ""
    logger.log(level, "%s done in %.2fs%s", name, time.perf_counter() - started, suffix)
