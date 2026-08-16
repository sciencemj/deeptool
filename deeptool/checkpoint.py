"""Checkpoint saving and restoring, and best-validation-loss snapshots."""

import copy
import os
from pathlib import Path
from typing import Any, Literal

import torch


def atomic_save(payload: dict[str, Any], path: str | Path) -> None:
    """Write to a temporary file, then swap it into place atomically.

    Best snapshots overwrite the same path on every improvement. An interrupted
    write would destroy the best weights collected so far, so the swap has to be
    atomic. `os.replace` is atomic on both POSIX and Windows, and on failure the
    previous file is left untouched.

    Args:
        payload: Anything `torch.save` accepts.
        path: Destination. A sibling `<path>.tmp` is used during the write.
    """
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def checkpoint_payload(model: torch.nn.Module, optim: torch.optim.Optimizer,
                       epoch: int, *, step: int | None = None
                       ) -> dict[str, Any]:
    """Build a full checkpoint payload that can resume training.

    Args:
        model: Model whose `state_dict` is stored.
        optim: Optimizer whose `state_dict` is stored.
        epoch: Epoch index to record.

    Returns:
        A dict with `model`, `optim`, `epoch` and `hparams` keys.
    """
    payload = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "hparams": getattr(model, "hparams", {}),
    }
    if step is not None:
        payload["step"] = step
    return payload


def save_checkpoint(model: torch.nn.Module, optim: torch.optim.Optimizer,
                    epoch: int, path: str | Path, *,
                    step: int | None = None) -> None:
    """Save model and optimizer state, epoch and hyperparameters to one file.

    Args:
        model: Model to save.
        optim: Optimizer to save.
        epoch: Epoch index to record.
        path: Destination file.
    """
    torch.save(checkpoint_payload(model, optim, epoch, step=step), path)


def load_checkpoint(path: str | Path, model: torch.nn.Module,
                    optim: torch.optim.Optimizer | None = None) -> dict[str, Any]:
    """Restore a checkpoint into `model` in place.

    Warning:
        `hparams` can hold arbitrary Python objects, so this reads with
        `weights_only=False`. Only load checkpoints you trust.

    Args:
        path: Checkpoint file.
        model: Model to restore into.
        optim: Optimizer to restore as well, for resuming training. Leave it out
            to restore weights only, for inference.

    Returns:
        A dict with the stored `epoch` and `hparams`.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optim is not None:
        optim.load_state_dict(ckpt["optim"])
    metadata = {"hparams": ckpt["hparams"]}
    for progress_name in ("epoch", "step"):
        if progress_name in ckpt:
            metadata[progress_name] = ckpt[progress_name]
    return metadata


class BestSnapshot:
    """Keeps model weights from the best value of one monitored scalar.

    With `path` unset the snapshot lives in memory as a `deepcopy`; with a path
    it is written to that file.

    Score and progress are tracked even when `enabled` is False. Disabling only
    skips the copy or write.

    Attributes:
        score: Best monitored score, or `None` before the first update.
        progress_name: Whether `progress` represents an epoch or step.
        progress: Epoch or step that produced the best score.
    """

    def __init__(self, enabled: bool = True, path: str | Path | None = None,
                 with_optim: bool = False, monitor: str = "val_loss",
                 mode: Literal["min", "max"] = "min") -> None:
        self.enabled = enabled
        self.path = path
        self.with_optim = with_optim
        self.monitor = monitor
        self.mode = mode
        self.score = None
        self.progress_name = None
        self.progress = None
        self._state = None

    def update(self, score: float,
               progress_name: Literal["epoch", "step"], progress: int,
               model: torch.nn.Module,
               optim: torch.optim.Optimizer) -> bool:
        """Record an improved score and snapshot the weights.

        Does nothing when `score` is not strictly better in the configured
        direction.

        Args:
            score: Finite monitored value for this progress point.
            progress_name: Whether `progress` is an epoch or step.
            progress: Epoch or step stored when this is a new best.
            model: Model whose `state_dict` is snapshotted.
            optim: Optimizer, used only when `with_optim` is set.

        Returns:
            `True` when the score improved, otherwise `False`.
        """
        improved = (
            self.score is None
            or (self.mode == "min" and score < self.score)
            or (self.mode == "max" and score > self.score)
        )
        if not improved:
            return False
        self.score = score
        self.progress_name = progress_name
        self.progress = progress
        if not self.enabled:
            return True
        if self.path is None:
            self._state = copy.deepcopy(model.state_dict())
            return True
        # optimizer 상태는 restore() 가 읽지 않는다. Adam 기준 모델의 2배라
        # 매 개선마다 쓰면 낭비이므로 기본값은 가중치 전용이다.
        if self.with_optim:
            payload = checkpoint_payload(model, optim, progress)
            if progress_name == "step":
                payload["step"] = payload.pop("epoch")
            if self.monitor != "val_loss" or progress_name != "epoch":
                payload.update({
                    "monitor": self.monitor,
                    "mode": self.mode,
                    "score": score,
                })
        elif self.monitor != "val_loss" or progress_name != "epoch":
            payload = {
                "model": model.state_dict(),
                progress_name: progress,
                "monitor": self.monitor,
                "mode": self.mode,
                "score": score,
            }
        else:
            payload = {
                "model": model.state_dict(),
                "epoch": progress,
                "val_loss": score,
            }
        atomic_save(payload, self.path)
        return True

    def restore(self, model: torch.nn.Module) -> int:
        """Restore `model` to the best weights and return its progress value.

        Only model weights are restored; optimizer state is left alone. The
        point is to evaluate with the best model, not to resume training.

        Args:
            model: Model to restore into.

        Returns:
            The epoch or step value that was restored.

        Raises:
            RuntimeError: If no snapshot exists — either there was no validation
                data, or snapshotting was disabled.
        """
        if self.progress is None:
            raise RuntimeError("No snapshot: there was no validation data.")
        if not self.enabled:
            raise RuntimeError(
                f"No snapshot: trained with snapshot_best=False. "
                f"(best was {self.progress_name} {self.progress}, "
                f"{self.monitor} {self.score:.4f})"
            )
        if self.path is None:
            model.load_state_dict(self._state)
        else:
            ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
        return self.progress
