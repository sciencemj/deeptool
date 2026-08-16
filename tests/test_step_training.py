import json

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data as torch_data

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


class StepMetricModel(CountingModel):
    def __init__(self, scores):
        super().__init__()
        self.scores = iter(scores)

    def validation_step(self, batch):
        loss = super().validation_step(batch)
        self.log("iou", next(self.scores))
        return loss


class ScheduledModel(CountingModel):
    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=1, gamma=0.1
        )
        return optim, scheduler


class PlateauStepModel(CountingModel):
    def __init__(self, losses):
        super().__init__()
        self.losses = iter(losses)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=0, factor=0.1
        )
        return optim, scheduler

    def validation_step(self, batch):
        return torch.tensor(next(self.losses), device=batch[0].device)


class EmptyData(DataModule):
    def get_dataloader(self, train):
        if not train:
            return None
        dataset = torch_data.TensorDataset(
            torch.empty(0, 1), torch.empty(0, 1)
        )
        return torch_data.DataLoader(dataset, batch_size=1, shuffle=False)


class LengthlessLoader:
    def __iter__(self):
        batch = (torch.zeros(1, 1), torch.zeros(1, 1))
        return iter([batch])


class LengthlessData(DataModule):
    def get_dataloader(self, train):
        return LengthlessLoader() if train else None


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


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


def test_step_trainer_uses_steps_for_board_and_plot_coordinates():
    trainer = Trainer(max_steps=3, plot=True)
    trainer.global_step = 2

    assert trainer.board.xlabel == "step"
    assert trainer.plot_x(train=True) == 3.0
    assert trainer.plot_x(train=False) == 2.0


def test_step_mode_performs_exactly_max_steps_updates():
    model = CountingModel()
    trainer = Trainer(max_steps=5, device="cpu", plot=False)

    trainer.fit(model, TinyData(train_batches=2, val_batches=0))

    assert model.updates == 5
    assert trainer.global_step == 5


def test_step_mode_repeats_loader_and_records_log_boundaries(tmp_path):
    trainer = Trainer(
        max_steps=5,
        log_every_n_steps=2,
        device="cpu",
        plot=False,
        log_dir=tmp_path / "exp",
    )

    trainer.fit(CountingModel(), TinyData(train_batches=2, val_batches=0))

    rows = _rows(tmp_path / "exp/history.jsonl")
    assert [row["step"] for row in rows] == [2, 4, 5]
    assert trainer.history["step"] == [2, 4, 5]
    assert len(trainer.history["train_loss"]) == 3
    assert all("epoch" not in row for row in rows)


def test_validation_boundary_forces_one_row_without_duplicates(tmp_path):
    trainer = Trainer(
        max_steps=6,
        log_every_n_steps=4,
        val_every_n_steps=3,
        device="cpu",
        plot=False,
        log_dir=tmp_path / "exp",
    )

    trainer.fit(CountingModel(), TinyData())

    rows = _rows(tmp_path / "exp/history.jsonl")
    assert [row["step"] for row in rows] == [3, 4, 6]
    assert ["val_loss" in row for row in rows] == [True, False, True]


def test_no_validation_interval_runs_validation_only_at_final_step(tmp_path):
    trainer = Trainer(
        max_steps=3,
        device="cpu",
        plot=False,
        log_dir=tmp_path / "exp",
    )

    trainer.fit(CountingModel(), TinyData())

    rows = _rows(tmp_path / "exp/history.jsonl")
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert ["val_loss" in row for row in rows] == [False, False, True]


def test_step_mode_rejects_empty_train_loader():
    with pytest.raises(ValueError, match="train dataloader.*empty"):
        Trainer(max_steps=1, plot=False).fit(CountingModel(), EmptyData())


def test_step_mode_rejects_lengthless_train_loader():
    with pytest.raises(ValueError, match="finite.*length"):
        Trainer(max_steps=1, plot=False).fit(CountingModel(), LengthlessData())


def test_step_monitor_tracks_best_step_and_restore_value():
    model = StepMetricModel([0.4, 0.8, 0.6])
    trainer = Trainer(
        max_steps=6,
        val_every_n_steps=2,
        monitor="iou",
        mode="max",
        device="cpu",
        plot=False,
    )

    trainer.fit(model, TinyData())

    assert trainer.best_score == pytest.approx(0.8)
    assert trainer.best_step == 4
    assert trainer.best_epoch is None
    assert trainer.restore_best() == 4


def test_step_patience_counts_validation_checks_not_steps():
    model = StepMetricModel([0.8, 0.7, 0.6, 0.9])
    trainer = Trainer(
        max_steps=10,
        val_every_n_steps=2,
        patience=2,
        monitor="iou",
        mode="max",
        device="cpu",
        plot=False,
    )

    trainer.fit(model, TinyData())

    assert trainer.global_step == 6
    assert trainer.history["step"][-1] == 6
    assert trainer.best_step == 2


def test_auto_scheduler_steps_after_each_update_in_step_mode(tmp_path):
    trainer = Trainer(
        max_steps=3,
        device="cpu",
        plot=False,
        log_dir=tmp_path / "exp",
    )

    trainer.fit(ScheduledModel(), TinyData(val_batches=0))

    rows = _rows(tmp_path / "exp/history.jsonl")
    assert [row["lr"] for row in rows] == pytest.approx(
        [0.1, 0.01, 0.001]
    )


def test_epoch_interval_scheduler_steps_when_loader_is_exhausted():
    trainer = Trainer(
        max_steps=5,
        scheduler_interval="epoch",
        device="cpu",
        plot=False,
    )

    trainer.fit(ScheduledModel(), TinyData(train_batches=2, val_batches=0))

    assert trainer.optim.param_groups[0]["lr"] == pytest.approx(0.001)


def test_plateau_scheduler_steps_on_step_validation_boundaries(tmp_path):
    trainer = Trainer(
        max_steps=6,
        log_every_n_steps=2,
        val_every_n_steps=2,
        device="cpu",
        plot=False,
        log_dir=tmp_path / "exp",
    )

    trainer.fit(PlateauStepModel([0.5, 0.7, 0.8]), TinyData())

    rows = _rows(tmp_path / "exp/history.jsonl")
    assert [row["lr"] for row in rows] == pytest.approx([0.1, 0.1, 0.01])


def test_step_checkpoint_includes_and_returns_global_step(tmp_path):
    trainer = Trainer(max_steps=3, device="cpu", plot=False)
    trainer.fit(CountingModel(), TinyData(val_batches=0))
    path = tmp_path / "checkpoint.pt"

    trainer.save_checkpoint(path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["step"] == 3
    meta = Trainer.load_checkpoint(path, CountingModel())
    assert meta["step"] == 3
