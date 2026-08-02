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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def plot(self, key: str, value: torch.Tensor | float, train: bool) -> None:
        """Draw one scalar on the live board. A no-op when there is no board.

        Args:
            key: Curve name. Rendered as `train_<key>` or `val_<key>`.
            value: A scalar tensor or plain float.
            train: Selects the training or validation curve. Training points use
                a fractional epoch on the x-axis; validation points use the
                epoch number.
        """
        if self.board is None or self.trainer is None:
            return
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        if train:
            x = self.trainer.train_batch_idx / self.trainer.num_train_batches
            every_n = self.trainer.num_train_batches / self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
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
