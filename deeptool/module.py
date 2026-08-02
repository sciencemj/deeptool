"""모델 규약 — nn.Module 에 하이퍼파라미터 저장과 플롯 훅을 얹는다."""

import torch
from torch import nn

from deeptool.core import HyperParameters


class Module(nn.Module, HyperParameters):
    """유저 모델의 베이스 클래스.

    유저는 ``forward``(또는 ``self.net``), ``loss``, ``configure_optimizers``
    셋만 채우면 된다. 노트북에서는 ``@add_to_class`` 로 나중 셀에서 붙여도 된다.

    ``board`` 와 ``trainer`` 는 ``Trainer.fit`` 이 주입한다.
    """

    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = None
        self.trainer = None

    def forward(self, X):
        assert hasattr(self, "net"), "forward() 를 구현하거나 self.net 을 정의하라"
        return self.net(X)

    def loss(self, y_hat, y):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def plot(self, key, value, train):
        """스칼라 하나를 보드에 찍는다. 보드가 없으면 아무것도 안 한다."""
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

    def training_step(self, batch):
        loss = self.loss(self(*batch[:-1]), batch[-1])
        self.plot("loss", loss, train=True)
        return loss

    def validation_step(self, batch):
        loss = self.loss(self(*batch[:-1]), batch[-1])
        self.plot("loss", loss, train=False)
        return loss
