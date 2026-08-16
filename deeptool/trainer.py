"""Training loop: device placement, epochs, loss aggregation, early stopping."""

from collections.abc import Sequence
from datetime import UTC, datetime
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import torch

from deeptool.board import ProgressBoard, _in_notebook
from deeptool.checkpoint import BestSnapshot
from deeptool.core import HyperParameters
from deeptool.data import DataModule
from deeptool.evaluate import Predictions
from deeptool.module import Module
from deeptool.record import RunRecorder
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
        snapshot_best: Keeps weights from the best monitored progress point.
        best_path: Writes that snapshot to this file instead of memory.
        best_with_optim: Also stores optimizer state in the snapshot file, so it
            can resume training.
        patience: Stops after this many epochs without improvement. Requires
            validation data.
        log_dir: Writes run metadata and one flushed JSONL row per epoch here.
        monitor: Exact scalar key used for best snapshots and early stopping.
        mode: Whether lower (`min`) or higher (`max`) monitor values improve.

    Raises:
        ValueError: If `patience` is below 1.
    """

    _ROW_FIELDS = frozenset({"epoch", "train_loss", "val_loss", "lr", "sec"})

    def __init__(self, max_epochs: int | None = None,
                 device: torch.device | str | None = None,
                 gradient_clip_val: float = 0, plot: bool = True,
                 snapshot_best: bool = True,
                 best_path: str | Path | None = None,
                 best_with_optim: bool = False,
                 patience: int | None = None,
                 log_dir: str | Path | None = None,
                 monitor: str = "val_loss",
                 mode: Literal["min", "max"] = "min",
                 max_steps: int | None = None,
                 log_every_n_steps: int = 1,
                 val_every_n_steps: int | None = None,
                 scheduler_interval: Literal[
                     "auto", "epoch", "step"
                 ] = "auto") -> None:
        self.save_hyperparameters()
        if (max_epochs is None) == (max_steps is None):
            raise ValueError("exactly one of max_epochs and max_steps is required")
        if max_epochs is not None:
            _require_positive_int("max_epochs", max_epochs)
        if max_steps is not None:
            _require_positive_int("max_steps", max_steps)
        _require_positive_int("log_every_n_steps", log_every_n_steps)
        if val_every_n_steps is not None:
            _require_positive_int("val_every_n_steps", val_every_n_steps)
        if scheduler_interval not in {"auto", "epoch", "step"}:
            raise ValueError(
                "scheduler_interval must be 'auto', 'epoch', or 'step'"
            )
        # patience=0 이면 최저점 epoch 에서도 epoch - best_epoch >= 0 이 참이 되어
        # 첫 epoch 직후 멈춘다. 의미가 없으므로 막는다.
        if patience is not None and patience < 1:
            raise ValueError(f"patience must be at least 1 (got {patience})")
        if not isinstance(monitor, str) or not monitor:
            raise ValueError("monitor must be a non-empty string")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.device = torch.device(device) if device is not None else default_device()
        self.training_unit = "epoch" if max_epochs is not None else "step"
        self.board = (
            ProgressBoard(
                xlabel=self.training_unit,
                ylabel="loss",
                display=_in_notebook(),
            )
            if plot else None
        )
        self.recorder = RunRecorder(log_dir) if log_dir is not None else None
        self.history = (
            {"train_loss": [], "val_loss": []}
            if self.training_unit == "epoch"
            else {"step": [], "train_loss": [], "val_loss": []}
        )
        self.global_step = 0
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self._epoch_scalars: dict[str, list[float]] = {}
        self._logged_metric_names: list[str] = []
        self._bad_monitor_checks = 0
        self._best_val_loss: float | None = None
        self._best = BestSnapshot(
            snapshot_best, best_path, best_with_optim,
            monitor=monitor, mode=mode,
        )

    @property
    def best_val_loss(self) -> float | None:
        """Lowest validation loss seen, or `None` before the first epoch."""
        return self._best_val_loss

    @property
    def best_score(self) -> float | None:
        """Best value seen for the configured monitor."""
        return self._best.score

    @property
    def best_epoch(self) -> int | None:
        """Epoch that produced the best score, or `None` in step mode."""
        if self._best.progress_name == "epoch":
            return self._best.progress
        return None

    @property
    def best_step(self) -> int | None:
        """Step that produced the best score, or `None` in epoch mode."""
        if self._best.progress_name == "step":
            return self._best.progress
        return None

    def prepare_data(self, data: DataModule) -> None:
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        try:
            self.num_train_batches = len(self.train_dataloader)
        except TypeError as error:
            if self.training_unit == "step":
                raise ValueError(
                    "step training needs a finite dataloader with a length"
                ) from error
            raise
        if self.training_unit == "step" and self.num_train_batches == 0:
            raise ValueError("train dataloader must not be empty")
        self.num_val_batches = (
            len(self.val_dataloader) if self.val_dataloader is not None else 0
        )

    def prepare_model(self, model: Module) -> None:
        model.trainer = self
        model.board = self.board
        self.model = model.to(self.device)

    def prepare_batch(self, batch: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return [a.to(self.device) for a in batch]

    def plot_x(self, train: bool) -> float:
        """Return the live-plot coordinate for the active training unit."""
        if self.training_unit == "step":
            return float(self.global_step + 1 if train else self.global_step)
        if train:
            return self.train_batch_idx / self.num_train_batches
        return float(self.epoch + 1)

    def materialize_lazy_parameters(self) -> None:
        """Materialize lazy layers with a dummy forward pass.

        `nn.LazyLinear` and friends have no parameters until the first forward,
        so building an optimizer before one would fail.
        """
        batch = self.prepare_batch(next(iter(self.train_dataloader)))
        with torch.no_grad():
            self.model(*batch[:-1])

    def fit(self, model: Module,
            data: DataModule) -> dict[str, list[float | None]]:
        self.prepare_data(data)
        # 검증 데이터가 없으면 best_epoch 가 계속 None 이라 조기 종료가 영원히
        # 발동하지 않는다. 조용히 무시하면 왜 안 멈추는지 알 수 없으므로 막는다.
        if (
            self.patience is not None
            and self.monitor == "val_loss"
            and self.num_val_batches == 0
        ):
            raise ValueError("patience needs validation data.")
        self.prepare_model(model)
        self.materialize_lazy_parameters()
        self._configure_optimizers()
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau
        if isinstance(self.scheduler, plateau) and self.num_val_batches == 0:
            raise ValueError("ReduceLROnPlateau needs validation data.")
        self._record_meta()
        if self.training_unit == "step":
            self._fit_steps()
            return self.history
        for self.epoch in range(self.max_epochs):
            started = perf_counter()
            self._epoch_scalars = {}
            self.fit_epoch()
            metrics = self._finish_logged_scalars()
            row = self._epoch_row(metrics, perf_counter() - started)
            improved = self._update_best(row, "epoch", self.epoch)
            self._record_row(row)
            self._step_scheduler()
            if self._should_stop_early(improved):
                break
        return self.history

    def _fit_steps(self) -> None:
        self.model.train()
        train_iterator = iter(self.train_dataloader)
        losses: list[float] = []
        last_lr = float(self.optim.param_groups[0]["lr"])
        window_started = perf_counter()
        self._epoch_scalars = {}
        while self.global_step < self.max_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                self.epoch += 1
                if self._resolved_scheduler_interval() == "epoch":
                    self._step_normal_scheduler()
                train_iterator = iter(self.train_dataloader)
                batch = next(train_iterator)
            loss, last_lr = self._train_batch(batch)
            losses.append(loss)
            validate = (
                self.global_step == self.max_steps
                or self.val_every_n_steps is not None
                and self.global_step % self.val_every_n_steps == 0
            )
            val_loss = self._run_validation() if validate else None
            if val_loss is not None:
                self._track_val_loss(val_loss)
            emit = (
                validate
                or self.global_step % self.log_every_n_steps == 0
                or self.global_step == self.max_steps
            )
            if not emit:
                continue
            metrics = self._finish_logged_scalars(append_history=False)
            row = self._step_row(
                losses,
                val_loss,
                metrics,
                last_lr,
                perf_counter() - window_started,
            )
            improved = (
                self._update_best(row, "step", self.global_step)
                if validate else None
            )
            self._append_step_history(row)
            self._record_row(row)
            if validate:
                self._step_plateau_scheduler(row)
            if validate and self._should_stop_early(improved):
                return
            losses = []
            self._epoch_scalars = {}
            window_started = perf_counter()
            self.model.train()

    def _configure_optimizers(self) -> None:
        configured = self.model.configure_optimizers()
        if isinstance(configured, tuple):
            if len(configured) != 2:
                raise ValueError(
                    "configure_optimizers must return an optimizer or an "
                    "(optimizer, scheduler) pair"
                )
            self.optim, self.scheduler = configured
        else:
            self.optim = configured
            self.scheduler = None

    def _step_scheduler(self) -> None:
        if isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            self.scheduler.step(self.history["val_loss"][-1])
        elif self._resolved_scheduler_interval() == "epoch":
            self._step_normal_scheduler()

    def _resolved_scheduler_interval(self) -> Literal["epoch", "step"]:
        if self.scheduler_interval == "auto":
            return self.training_unit
        return self.scheduler_interval

    def _step_normal_scheduler(self) -> None:
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau
        if self.scheduler is not None and not isinstance(self.scheduler, plateau):
            self.scheduler.step()

    def _step_plateau_scheduler(self, row: dict[str, object]) -> None:
        if isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            self.scheduler.step(float(row["val_loss"]))

    def _record_meta(self) -> None:
        if self.recorder is None:
            return
        model_class = type(self.model)
        self.recorder.meta(
            started_at=datetime.now(UTC).isoformat(),
            device=str(self.device),
            model_class=f"{model_class.__module__}.{model_class.__qualname__}",
            trainer_hparams=self.hparams,
            model_hparams=getattr(self.model, "hparams", {}),
        )

    def _epoch_row(self, metrics: dict[str, float],
                   seconds: float) -> dict[str, object]:
        row: dict[str, object] = {
            "epoch": self.epoch,
            "train_loss": self.history["train_loss"][-1],
        }
        if self.num_val_batches > 0:
            row["val_loss"] = self.history["val_loss"][-1]
        row.update(metrics)
        row["lr"] = float(self.optim.param_groups[0]["lr"])
        row["sec"] = float(seconds)
        return row

    def _record_row(self, row: dict[str, object]) -> None:
        if self.recorder is None:
            return
        self.recorder.epoch(**row)
        if not _in_notebook():
            print(" ".join(
                f"{key}={_format_scalar(value)}" for key, value in row.items()
            ))

    def _update_best(
        self, row: dict[str, object],
        progress_name: Literal["epoch", "step"], progress: int,
    ) -> bool | None:
        if self.monitor == "val_loss" and "val_loss" not in row:
            return None
        if self.monitor not in row:
            raise ValueError(
                f"monitor {self.monitor!r} was not logged at "
                f"{progress_name} {progress}"
            )
        score = float(row[self.monitor])
        if not math.isfinite(score):
            raise ValueError(
                f"monitor {self.monitor!r} must be finite at "
                f"{progress_name} {progress}"
            )
        return self._best.update(
            score, progress_name, progress, self.model, self.optim
        )

    def _log_scalar(self, key: str, value: float) -> None:
        if key in self._ROW_FIELDS:
            raise ValueError(f"{key!r} is reserved for Trainer epoch rows")
        self._epoch_scalars.setdefault(key, []).append(value)

    def _finish_logged_scalars(
        self, *, append_history: bool = True
    ) -> dict[str, float]:
        metrics = {
            key: sum(values) / len(values)
            for key, values in self._epoch_scalars.items()
        }
        if append_history:
            self._append_epoch_metrics(metrics)
        return metrics

    def _append_epoch_metrics(self, metrics: dict[str, float]) -> None:
        for key in self._logged_metric_names:
            if key not in metrics:
                self.history[key].append(None)
        for key, value in metrics.items():
            if key not in self._logged_metric_names:
                self._logged_metric_names.append(key)
                self.history[key] = [None] * self.epoch
            self.history[key].append(value)

    def _step_row(
        self,
        train_losses: list[float],
        val_loss: float | None,
        metrics: dict[str, float],
        lr: float,
        seconds: float,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "step": self.global_step,
            "train_loss": sum(train_losses) / len(train_losses),
        }
        if val_loss is not None:
            row["val_loss"] = val_loss
        row.update(metrics)
        row["lr"] = lr
        row["sec"] = seconds
        return row

    def _append_step_history(self, row: dict[str, object]) -> None:
        history_keys = {
            key for key in row if key not in {"lr", "sec"}
        }
        previous = len(self.history["step"])
        for key in history_keys - self.history.keys():
            self.history[key] = [None] * previous
        for key in self.history:
            self.history[key].append(row.get(key))

    def fit_epoch(self) -> None:
        self.model.train()
        losses = []
        for batch in self.train_dataloader:
            loss, _ = self._train_batch(batch)
            losses.append(loss)
        self.history["train_loss"].append(sum(losses) / len(losses))

        val_loss = self._run_validation()
        if val_loss is None:
            return
        self.history["val_loss"].append(val_loss)
        self._track_val_loss(val_loss)

    def _run_validation(self) -> float | None:
        if self.num_val_batches == 0:
            return None
        self.model.eval()
        losses = []
        for batch in self.val_dataloader:
            with torch.no_grad():
                loss = self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1
            losses.append(loss.detach().cpu().item())
        return sum(losses) / len(losses)

    def _track_val_loss(self, val_loss: float) -> None:
        if self._best_val_loss is None or val_loss < self._best_val_loss:
            self._best_val_loss = val_loss

    def _train_batch(
        self, batch: Sequence[torch.Tensor]
    ) -> tuple[float, float]:
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        loss.backward()
        if self.gradient_clip_val > 0:
            self.clip_gradients(self.gradient_clip_val)
        lr = float(self.optim.param_groups[0]["lr"])
        self.optim.step()
        self.train_batch_idx += 1
        self.global_step += 1
        if self._resolved_scheduler_interval() == "step":
            self._step_normal_scheduler()
        return float(loss.detach().cpu().item()), lr

    def clip_gradients(self, grad_clip_val: float) -> None:
        params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, grad_clip_val)

    def _should_stop_early(self, improved: bool | None) -> bool:
        """True once `patience` monitor checks pass without improvement."""
        if self.patience is None or improved is None:
            return False
        self._bad_monitor_checks = (
            0 if improved else self._bad_monitor_checks + 1
        )
        return self._bad_monitor_checks >= self.patience

    def restore_best(self) -> int:
        """Load weights from the best value of the configured monitor.

        `fit` never does this on its own. Until you call it the model holds the
        last epoch's weights, so you can compare the two.

        Only model weights are restored; optimizer state is left alone.

        Returns:
            The epoch or step value that was restored.

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


def _format_scalar(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _require_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
