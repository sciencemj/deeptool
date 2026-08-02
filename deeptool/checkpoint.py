"""Checkpoint saving and restoring, and best-validation-loss snapshots."""

import copy
import os

import torch


def atomic_save(payload, path):
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


def checkpoint_payload(model, optim, epoch):
    """Build a full checkpoint payload that can resume training.

    Args:
        model: Model whose `state_dict` is stored.
        optim: Optimizer whose `state_dict` is stored.
        epoch: Epoch index to record.

    Returns:
        A dict with `model`, `optim`, `epoch` and `hparams` keys.
    """
    return {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "hparams": getattr(model, "hparams", {}),
    }


def save_checkpoint(model, optim, epoch, path):
    """Save model and optimizer state, epoch and hyperparameters to one file.

    Args:
        model: Model to save.
        optim: Optimizer to save.
        epoch: Epoch index to record.
        path: Destination file.
    """
    torch.save(checkpoint_payload(model, optim, epoch), path)


def load_checkpoint(path, model, optim=None):
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
    return {"epoch": ckpt["epoch"], "hparams": ckpt["hparams"]}


class BestSnapshot:
    """Keeps the model weights from the epoch with the lowest validation loss.

    With `path` unset the snapshot lives in memory as a `deepcopy`; with a path
    it is written to that file.

    `val_loss` and `epoch` are tracked even when `enabled` is False. Comparing
    floats costs nothing, and knowing which epoch was best is useful on its own.
    Disabling only skips the copy or the write.

    Attributes:
        val_loss: Lowest validation loss seen, or `None` before the first epoch.
        epoch: Epoch that produced it, or `None`.
    """

    def __init__(self, enabled=True, path=None, with_optim=False):
        self.enabled = enabled
        self.path = path
        self.with_optim = with_optim
        self.val_loss = None
        self.epoch = None
        self._state = None

    def update(self, val_loss, epoch, model, optim):
        """Record a new minimum and snapshot the weights.

        Does nothing when `val_loss` is not lower than the current best.

        Args:
            val_loss: Mean validation loss for this epoch.
            epoch: Epoch index, stored when this is a new best.
            model: Model whose `state_dict` is snapshotted.
            optim: Optimizer, used only when `with_optim` is set.
        """
        if self.val_loss is not None and val_loss >= self.val_loss:
            return
        self.val_loss = val_loss
        self.epoch = epoch
        if not self.enabled:
            return
        if self.path is None:
            self._state = copy.deepcopy(model.state_dict())
            return
        # optimizer 상태는 restore() 가 읽지 않는다. Adam 기준 모델의 2배라
        # 매 개선마다 쓰면 낭비이므로 기본값은 가중치 전용이다.
        if self.with_optim:
            payload = checkpoint_payload(model, optim, epoch)
        else:
            payload = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }
        atomic_save(payload, self.path)

    def restore(self, model):
        """Restore `model` to the best weights and return that epoch.

        Only model weights are restored; optimizer state is left alone. The
        point is to evaluate with the best model, not to resume training.

        Args:
            model: Model to restore into.

        Returns:
            The epoch index that was restored.

        Raises:
            RuntimeError: If no snapshot exists — either there was no validation
                data, or snapshotting was disabled.
        """
        if self.epoch is None:
            raise RuntimeError("No snapshot: there was no validation data.")
        if not self.enabled:
            raise RuntimeError(
                f"No snapshot: trained with snapshot_best=False. "
                f"(best was epoch {self.epoch}, "
                f"val_loss {self.val_loss:.4f})"
            )
        if self.path is None:
            model.load_state_dict(self._state)
        else:
            ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
        return self.epoch
