"""Model contract: nn.Module plus hyperparameter capture and plotting hooks."""

from collections.abc import Sequence

import torch
from torch import nn

from deeptool.core import HyperParameters


class Module(nn.Module, HyperParameters):
    """Base class for your models.

    You fill in three things: `forward` (or just assign `self.net`), `loss`, and
    `configure_optimizers`. In a notebook you can attach them from later cells
    with `@add_to_class`.

    `board` and `trainer` are injected by `Trainer.fit`.
    """

    def __init__(self, plot_train_per_epoch: int = 2,
                 plot_valid_per_epoch: int = 1) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.board = None
        self.trainer = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "net"), "implement forward() or assign self.net"
        return self.net(X)

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def configure_optimizers(
        self,
    ) -> (
        torch.optim.Optimizer
        | tuple[
            torch.optim.Optimizer,
            torch.optim.lr_scheduler.LRScheduler
            | torch.optim.lr_scheduler.ReduceLROnPlateau,
        ]
    ):
        """Build an optimizer and, optionally, a learning-rate scheduler.

        Returns:
            An optimizer, or `(optimizer, scheduler)`. Normal scheduler timing
            follows `Trainer.scheduler_interval`; `ReduceLROnPlateau` receives
            validation loss whenever validation runs.
        """
        raise NotImplementedError

    def log(self, key: str, value: torch.Tensor | float) -> None:
        """Aggregate one custom scalar under its unchanged name for this interval.

        When a live board is attached, the same value is also plotted using the
        model's current training or validation phase. Calling this method before
        a Trainer is attached is a no-op.

        Args:
            key: Free-form metric name stored in progress history.
            value: A scalar tensor or plain number.

        Raises:
            ValueError: If `value` is not scalar or `key` is reserved by Trainer.
        """
        if self.trainer is None:
            return
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError("logged tensors must contain one scalar value")
            value = value.detach().cpu().item()
        scalar = float(value)
        self.trainer._log_scalar(key, scalar)
        self.plot(key, scalar, train=self.training)

    def plot(self, key: str, value: torch.Tensor | float, train: bool) -> None:
        """Draw one scalar on the live board. A no-op when there is no board.

        Args:
            key: Curve name. Rendered as `train_<key>` or `val_<key>`.
            value: A scalar tensor or plain float.
            train: Selects the training or validation curve. Training points use
                the active Trainer progress unit on the x-axis.
        """
        if self.board is None or self.trainer is None:
            return
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        x = self.trainer.plot_x(train)
        if train:
            every_n = self.trainer.num_train_batches / self.plot_train_per_epoch
        else:
            every_n = self.trainer.num_val_batches / self.plot_valid_per_epoch
        prefix = "train_" if train else "val_"
        self.board.draw(x, float(value), prefix + key,
                        every_n=max(1, int(every_n)))

    def training_step(self, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        loss = self.loss(self(*batch[:-1]), batch[-1])
        self.plot("loss", loss, train=True)
        return loss

    def validation_step(self, batch: Sequence[torch.Tensor]) -> torch.Tensor:
        loss = self.loss(self(*batch[:-1]), batch[-1])
        self.plot("loss", loss, train=False)
        return loss
