"""Uniform logger setup writing to logs/<phase>-<UTCstamp>.log."""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime

from llm_textcls.io import project_root


def get_logger(name: str, phase: str) -> logging.Logger:
    """Return a logger that writes to a phase-specific UTC-stamped log file.

    File handler captures DEBUG+; stderr handler captures WARNING+ so progress
    bars and routine output stay clean.

    Args:
        name: Logger name (usually ``__name__``).
        phase: Phase label used in log filename (e.g., ``"phase1"``).

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    log_dir = project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{phase}-{ts}.log"

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.WARNING)
    sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logger.addHandler(sh)

    logger.debug("Log file: %s", log_path)
    return logger
