"""Evaluation: the metric suite per group, the eight test protocols, and Phase-0 falsification."""
from .evaluate import evaluate_per_group, evaluate_scores
from .falsify import run_falsification
from .protocols import PROTOCOLS

__all__ = ["evaluate_scores", "evaluate_per_group", "run_falsification", "PROTOCOLS"]
