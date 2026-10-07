"""Training: losses and the loop. Observability first -- log hparams, seeds, shapes, metrics."""
from .losses import cllr_loss, detection_loss

__all__ = ["cllr_loss", "detection_loss"]
