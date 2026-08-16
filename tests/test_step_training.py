import pytest
import torch
from torch import nn
from torch.nn import functional as F

from deeptool.data import DataModule
from deeptool.module import Module
from deeptool.trainer import Trainer


class TinyData(DataModule):
    def __init__(self, train_batches=2, val_batches=1):
        super().__init__(batch_size=1)
        self.X = torch.arange(
            train_batches + val_batches, dtype=torch.float32
        ).view(-1, 1)
        self.y = 2 * self.X
        self.train_batches = train_batches
        self.val_batches = val_batches

    def get_dataloader(self, train):
        if train:
            return self.get_tensorloader(
                (self.X, self.y), True, slice(0, self.train_batches)
            )
        if self.val_batches == 0:
            return None
        return self.get_tensorloader(
            (self.X, self.y),
            False,
            slice(
                self.train_batches,
                self.train_batches + self.val_batches,
            ),
        )


class CountingModel(Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1, 1)
        self.updates = 0

    def loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def training_step(self, batch):
        self.updates += 1
        return super().training_step(batch)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one"),
        ({"max_epochs": 1, "max_steps": 2}, "exactly one"),
        ({"max_steps": 0}, "max_steps"),
        ({"max_steps": True}, "max_steps"),
        ({"max_epochs": False}, "max_epochs"),
        ({"max_steps": 2, "log_every_n_steps": 0}, "log_every"),
        ({"max_steps": 2, "val_every_n_steps": 0}, "val_every"),
        ({"max_steps": 2, "scheduler_interval": "batch"}, "scheduler_interval"),
    ],
)
def test_training_limit_and_interval_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Trainer(**kwargs)


def test_training_unit_is_derived_from_the_selected_limit():
    assert Trainer(max_epochs=2).training_unit == "epoch"

    trainer = Trainer(max_steps=3)

    assert trainer.training_unit == "step"
    assert trainer.global_step == 0


def test_step_mode_performs_exactly_max_steps_updates():
    model = CountingModel()
    trainer = Trainer(max_steps=5, device="cpu", plot=False)

    trainer.fit(model, TinyData(train_batches=2, val_batches=0))

    assert model.updates == 5
    assert trainer.global_step == 5
