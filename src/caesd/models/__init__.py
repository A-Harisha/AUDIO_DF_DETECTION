"""Model blocks. One shared condition vector steers evidence (FiLM) and calibration (hypernet)."""
from .backbone import Backbone
from .block_a_condition import ConditionEncoder
from .block_b_sae import SparseAutoencoder
from .block_c_hypernet import CalibrationHypernet
from .block_d_conformal import MondrianConformalGate
from .detector import CaesdDetector, DetectorOutput
from .film import FiLM

__all__ = [
    "Backbone",
    "CaesdDetector",
    "CalibrationHypernet",
    "ConditionEncoder",
    "DetectorOutput",
    "FiLM",
    "MondrianConformalGate",
    "SparseAutoencoder",
]
