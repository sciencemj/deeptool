"""Training loop: device placement, epochs, loss aggregation, early stopping."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from deeptool.board import ProgressBoard
from deeptool.checkpoint import BestSnapshot
from deeptool.core import HyperParameters
from deeptool.data import DataModule
from deeptool.evaluate import Predictions
from deeptool.module import Module
# 아래 셋은 Trainer 의 동명 메서드와 겹치므로 별칭으로 가져온다.
from deeptool.checkpoint import load_checkpoint as _load_checkpoint
from deeptool.checkpoint import save_checkpoint as _save_checkpoint
from deeptool.evaluate import predict as _predict


def default_device() -> torch.device:
    """Pick the first available accelerator, in the order cuda, mps, cpu.

    Returns:
        A `torch.device`.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer(HyperParameters):
    """Runs the training loop over a `Module` and a `DataModule`.

    All configuration lives on the constructor; `fit` takes only the model and
    the data. One `Trainer` therefore represents one training setup, and
    `trainer.hparams` records it in full.

    Args:
        max_epochs: Upper bound on epochs. Early stopping may end sooner.
        device: Where to train. Defaults to `default_device()`.
        gradient_clip_val: Clips gradient norm after backward when above zero.
        plot: Draws a live loss curve in the notebook.
        snapshot_best: Keeps the weights from the lowest-validation-loss epoch.
        best_path: Writes that snapshot to this file instead of memory.
        best_with_optim: Also stores optimizer state in the snapshot file, so it
            can resume training.
        patience: Stops after this many epochs without improvement. Requires
            validation data.

    Raises:
        ValueError: If `patience` is below 1.
    """

    def __init__(self, max_epochs: int,
                 device: torch.device | str | None = None,
                 gradient_clip_val: float = 0, plot: bool = True,
                 snapshot_best: bool = True,
                 best_path: str | Path | None = None,
                 best_with_optim: bool = False,
                 patience: int | None = None) -> None:
        self.save_hyperparameters()
        # patience=0 이면 최저점 epoch 에서도 epoch - best_epoch >= 0 이 참이 되어
        # 첫 epoch 직후 멈춘다. 의미가 없으므로 막는다.
        if patience is not None and patience < 1:
            raise ValueError(f"patience must be at least 1 (got {patience})")
        self.device = torch.device(device) if device is not None else default_device()
        self.board = ProgressBoard(xlabel="epoch", ylabel="loss") if plot else None
        self.history = {"train_loss": [], "val_loss": []}
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self._best = BestSnapshot(snapshot_best, best_path, best_with_optim)

    @property
    def best_val_loss(self) -> float | None:
        """Lowest validation loss seen, or `None` before the first epoch."""
        return self._best.val_loss

    @property
    def best_epoch(self) -> int | None:
        """Epoch that produced the lowest validation loss, or `None`."""
        return self._best.epoch

    def prepare_data(self, data: DataModule) -> None:
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (
            len(self.val_dataloader) if self.val_dataloader is not None else 0
        )

    def prepare_model(self, model: Module) -> None:
        model.trainer = self
        model.board = self.board
        self.model = model.to(self.device)

    def prepare_batch(self, batch: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [a.to(self.device) for a in batch]

    def materialize_lazy_parameters(self) -> None:
        """Materialize lazy layers with a dummy forward pass.

        `nn.LazyLinear` and friends have no parameters until the first forward,
        so building an optimizer before one would fail.
        """
        batch = self.prepare_batch(next(iter(self.train_dataloader)))
        with torch.no_grad():
            self.model(*batch[:-1])

    def fit(self, model: Module, data: DataModule) -> dict[str, list[float]]:
        self.prepare_data(data)
        # 검증 데이터가 없으면 best_epoch 가 계속 None 이라 조기 종료가 영원히
        # 발동하지 않는다. 조용히 무시하면 왜 안 멈추는지 알 수 없으므로 막는다.
        if self.patience is not None and self.num_val_batches == 0:
            raise ValueError("patience needs validation data.")
        self.prepare_model(model)
        self.materialize_lazy_parameters()
        self.optim = self.model.configure_optimizers()
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
            if self._should_stop_early():
                break
        return self.history

    def fit_epoch(self) -> None:
        self.model.train()
        losses = []
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            loss.backward()
            if self.gradient_clip_val > 0:
                self.clip_gradients(self.gradient_clip_val)
            self.optim.step()
            self.train_batch_idx += 1
            losses.append(loss.detach().cpu().item())
        self.history["train_loss"].append(sum(losses) / len(losses))

        if self.num_val_batches == 0:
            return
        self.model.eval()
        losses = []
        for batch in self.val_dataloader:
            with torch.no_grad():
                loss = self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1
            losses.append(loss.detach().cpu().item())
        val_loss = sum(losses) / len(losses)
        self.history["val_loss"].append(val_loss)
        self._best.update(val_loss, self.epoch, self.model, self.optim)

    def clip_gradients(self, grad_clip_val: float) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, grad_clip_val)

    def _should_stop_early(self) -> bool:
        """True once `patience` epochs have passed without improvement."""
        if self.patience is None or self.best_epoch is None:
            return False
        return self.epoch - self.best_epoch >= self.patience

    def restore_best(self) -> int:
        """Load the weights from the epoch with the lowest validation loss.

        `fit` never does this on its own. Until you call it the model holds the
        last epoch's weights, so you can compare the two.

        Only model weights are restored; optimizer state is left alone.

        Returns:
            The epoch index that was restored.

        Raises:
            RuntimeError: If `fit` has not run, if there was no validation data,
                or if `snapshot_best` was off.
        """
        if not hasattr(self, "model"):
            raise RuntimeError("fit() has not run yet.")
        return self._best.restore(self.model)

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model and optimizer state, epoch and hyperparameters to a file.

        Args:
            path: Destination file.
        """
        _save_checkpoint(self.model, self.optim, self.epoch, path)

    @staticmethod
    def load_checkpoint(path: str | Path, model: torch.nn.Module,
                        optim: torch.optim.Optimizer | None = None) -> dict[str, Any]:
        """Restore a checkpoint into `model` in place.

        Args:
            path: Checkpoint file.
            model: Model to restore into.
            optim: Optimizer to restore as well, for resuming training. Leave it
                out to restore weights only, for inference.

        Returns:
            A dict with the stored `epoch` and `hparams`.
        """
        return _load_checkpoint(path, model, optim)

    def predict(self, data: DataModule, train: bool = False,
                keep_inputs: bool = False) -> Predictions:
        """Run the trained model over `data` and collect per-sample results.

        Args:
            data: A `DataModule`.
            train: Uses the training split instead of validation.
            keep_inputs: Also collect the input tensors, for visualizing
                individual samples.

        Returns:
            A `Predictions` holding CPU tensors.
        """
        loader = data.train_dataloader() if train else data.val_dataloader()
        return _predict(self.model, loader, self.device, keep_inputs)
