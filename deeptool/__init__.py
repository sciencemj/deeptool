"""Object-oriented deep learning helpers for Jupyter notebooks.

You write the model in plain PyTorch. `deeptool` handles what surrounds it:
hyperparameter capture, attaching methods across notebook cells, live loss
curves, device selection, checkpoints and post-hoc evaluation.
"""

from deeptool.board import ProgressBoard
from deeptool.core import HyperParameters, add_to_class
from deeptool.data import DataModule
from deeptool.evaluate import Predictions, predict
from deeptool.module import Module
from deeptool.trainer import Trainer, default_device

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
