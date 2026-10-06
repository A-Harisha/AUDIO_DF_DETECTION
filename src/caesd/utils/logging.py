"""Logging helpers. Observability is a project rule: log hparams, seeds, metrics, shapes."""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_shapes(logger: logging.Logger, **tensors: Any) -> None:
    """Log the shape of each named tensor/array. Call at every block boundary."""
    for name, value in tensors.items():
        shape = tuple(getattr(value, "shape", ())) or type(value).__name__
        logger.info("shape | %s = %s", name, shape)


def log_hparams(logger: logging.Logger, hparams: Mapping[str, Any]) -> None:
    logger.info("hparams | %s", json.dumps(dict(hparams), default=str, sort_keys=True))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
