"""ood — object-oriented deep learning helpers for Jupyter notebooks."""

from ood.board import ProgressBoard
from ood.core import HyperParameters, add_to_class
from ood.data import DataModule
from ood.module import Module
from ood.trainer import Trainer, default_device

__version__ = "0.1.0"

__all__ = [
    "DataModule",
    "HyperParameters",
    "Module",
    "ProgressBoard",
    "Trainer",
    "add_to_class",
    "default_device",
    "__version__",
]
