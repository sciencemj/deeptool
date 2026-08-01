"""ood — object-oriented deep learning helpers for Jupyter notebooks."""

from ood.board import ProgressBoard
from ood.core import HyperParameters, add_to_class
from ood.data import DataModule

__version__ = "0.1.0"

__all__ = [
    "DataModule",
    "HyperParameters",
    "ProgressBoard",
    "add_to_class",
    "__version__",
]
