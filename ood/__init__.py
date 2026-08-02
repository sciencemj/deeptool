"""ood — object-oriented deep learning helpers for Jupyter notebooks."""

from ood.board import ProgressBoard
from ood.core import HyperParameters, add_to_class
from ood.data import DataModule
from ood.evaluate import Predictions, predict
from ood.module import Module
from ood.trainer import Trainer, default_device

__version__ = "0.2.0"

__all__ = [
    "DataModule",
    "HyperParameters",
    "Module",
    "Predictions",
    "ProgressBoard",
    "Trainer",
    "add_to_class",
    "default_device",
    "predict",
    "__version__",
]
