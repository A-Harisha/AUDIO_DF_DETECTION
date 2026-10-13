"""Model blocks: backbone, detector head, block A (condition), B (SAE), C (post-hoc calibration
hypernetwork), D (conformal gate)."""
from .backbone import Backbone
from .block_a_condition import ConditionEncoder
from .block_b_sae import SparseAutoencoder
from .block_c_hypernet import CalibrationHypernet, label_probe_eer
from .block_d_conformal import MondrianConformalGate
from .detector import CaesdDetector, Head
from .film import FiLM
from .raw_frontend import RawFrontEnd

__all__ = [
    "Backbone",
    "CaesdDetector",
    "CalibrationHypernet",
    "ConditionEncoder",
    "FiLM",
    "Head",
    "MondrianConformalGate",
    "RawFrontEnd",
    "SparseAutoencoder",
    "label_probe_eer",
]
