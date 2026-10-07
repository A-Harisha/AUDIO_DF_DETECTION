"""Published baselines run under our own protocol. AASIST anchors the setup to the literature."""
from .aasist import AasistScorer

__all__ = ["AasistScorer"]
