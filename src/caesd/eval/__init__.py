"""Evaluation: metric suite per group, the eight protocols, Phase-0 falsification, Phase-1 measurement."""
from .evaluate import evaluate_per_group, evaluate_scores
from .falsify import run_falsification
from .measure import run_measurement
from .protocols import PROTOCOLS

__all__ = ["PROTOCOLS", "evaluate_per_group", "evaluate_scores", "run_falsification", "run_measurement"]
