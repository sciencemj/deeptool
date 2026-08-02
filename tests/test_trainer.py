import pytest
import torch
from torch import nn
from torch.nn import functional as F

from deeptool.core import add_to_class
from deeptool.data import DataModule
from deeptool.module import Module
from deeptool.trainer import Trainer, default_device


class LinearData(DataModule):
    def __init__(self, n=64, batch_size=16):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(0)
        self.X = torch.randn(n, 2)
        self.y = self.X @ torch.tensor([[2.0], [-3.0]]) + 1.0

    def get_dataloader(self, train):
        idx = slice(0, 48) if train else slice(48, None)
        return self.get_tensorloader((self.X, self.y), train, idx)


class LinReg(Module):
    def __init__(self, lr=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Linear(2, 1)


@add_to_class(LinReg)
def loss(self, y_hat, y):
    return F.mse_loss(y_hat, y)


@add_to_class(LinReg)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)


class LazyLinReg(LinReg):
    def __init__(self, lr=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)


def test_default_device_returns_a_torch_device():
    assert isinstance(default_device(), torch.device)


def test_explicit_device_is_respected():
    assert Trainer(max_epochs=1, device="cpu").device == torch.device("cpu")


def test_fit_records_one_loss_per_epoch():
    trainer = Trainer(max_epochs=3, device="cpu", plot=False)
    history = trainer.fit(LinReg(), LinearData())

    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3


def test_fit_reduces_training_loss():
    torch.manual_seed(0)
    trainer = Trainer(max_epochs=15, device="cpu", plot=False)
    history = trainer.fit(LinReg(), LinearData())

    assert history["train_loss"][-1] < history["train_loss"][0]


def test_fit_advances_batch_counters():
    trainer = Trainer(max_epochs=2, device="cpu", plot=False)
    trainer.fit(LinReg(), LinearData())

    assert trainer.train_batch_idx == 2 * trainer.num_train_batches
    assert trainer.val_batch_idx == 2 * trainer.num_val_batches


def test_fit_materialises_lazy_parameters_before_optimizer():
    """LazyLinear 는 첫 forward 전까지 파라미터가 없어 optimizer 생성이 터진다."""
    trainer = Trainer(max_epochs=1, device="cpu", plot=False)
    history = trainer.fit(LazyLinReg(), LinearData())

    assert len(history["train_loss"]) == 1


def test_plot_true_populates_the_board():
    trainer = Trainer(max_epochs=2, device="cpu", plot=True)
    trainer.fit(LinReg(), LinearData())

    assert trainer.board.data["train_loss"]
    assert trainer.board.data["val_loss"]


def test_plot_false_leaves_no_board():
    trainer = Trainer(max_epochs=1, device="cpu", plot=False)
    trainer.fit(LinReg(), LinearData())

    assert trainer.board is None


def test_gradient_clipping_runs_without_error():
    trainer = Trainer(max_epochs=1, device="cpu", plot=False, gradient_clip_val=1.0)
    history = trainer.fit(LinReg(), LinearData())

    assert len(history["train_loss"]) == 1


def test_save_checkpoint_writes_a_loadable_file(tmp_path):
    trainer = Trainer(max_epochs=2, device="cpu", plot=False)
    trainer.fit(LinReg(), LinearData())
    path = tmp_path / "ckpt.pt"

    trainer.save_checkpoint(path)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert set(ckpt) == {"model", "optim", "epoch", "hparams"}


def test_load_checkpoint_restores_model_weights(tmp_path):
    trainer = Trainer(max_epochs=2, device="cpu", plot=False)
    trained = LinReg()
    trainer.fit(trained, LinearData())
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path)

    fresh = LinReg()
    Trainer.load_checkpoint(path, fresh)

    for a, b in zip(trained.state_dict().values(), fresh.state_dict().values()):
        assert torch.equal(a.cpu(), b)


def test_load_checkpoint_returns_epoch_and_hparams(tmp_path):
    trainer = Trainer(max_epochs=3, device="cpu", plot=False)
    trainer.fit(LinReg(lr=0.05), LinearData())
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path)

    meta = Trainer.load_checkpoint(path, LinReg())

    assert meta["epoch"] == 2  # 0-indexed, max_epochs=3 의 마지막
    assert meta["hparams"] == {"lr": 0.05}


def test_load_checkpoint_restores_optimizer_when_given(tmp_path):
    trainer = Trainer(max_epochs=2, device="cpu", plot=False)
    trainer.fit(LinReg(), LinearData())
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path)

    fresh = LinReg()
    optim = fresh.configure_optimizers()
    Trainer.load_checkpoint(path, fresh, optim)

    assert optim.state_dict()["param_groups"][0]["lr"] == 0.1


def test_load_checkpoint_without_optimizer_leaves_it_alone(tmp_path):
    trainer = Trainer(max_epochs=1, device="cpu", plot=False)
    trainer.fit(LinReg(), LinearData())
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path)

    meta = Trainer.load_checkpoint(path, LinReg(), optim=None)

    assert "epoch" in meta


class ScriptedData(DataModule):
    """검증 배치가 정확히 1개다. validation_step 호출 1회 = epoch 1회."""

    def __init__(self, batch_size=4):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(0)
        self.X = torch.randn(16, 2)
        self.y = self.X @ torch.tensor([[2.0], [-3.0]]) + 1.0

    def get_dataloader(self, train):
        idx = slice(0, 12) if train else slice(12, None)
        return self.get_tensorloader((self.X, self.y), train, idx)


class NoValData(DataModule):
    """검증 로더가 없다."""

    def __init__(self, batch_size=4):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(0)
        self.X = torch.randn(12, 2)
        self.y = self.X @ torch.tensor([[2.0], [-3.0]]) + 1.0

    def get_dataloader(self, train):
        if not train:
            return None
        return self.get_tensorloader((self.X, self.y), train, slice(0, None))


class ScriptedLoss(LinReg):
    """검증 손실을 미리 정한 수열대로 돌려준다.

    학습은 정상적으로 진행되므로 epoch 마다 가중치가 실제로 달라진다.
    """

    def __init__(self, losses, lr=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.call_count = 0


@add_to_class(ScriptedLoss)
def validation_step(self, batch):
    loss = torch.tensor(self.losses[self.call_count])
    self.call_count += 1
    return loss


def test_best_epoch_is_the_last_when_val_loss_decreases_monotonically():
    trainer = Trainer(max_epochs=4, device="cpu", plot=False)
    trainer.fit(ScriptedLoss([0.9, 0.7, 0.5, 0.3]), ScriptedData())

    assert trainer.best_epoch == 3


def test_best_epoch_points_at_the_minimum():
    trainer = Trainer(max_epochs=4, device="cpu", plot=False)
    trainer.fit(ScriptedLoss([0.5, 0.3, 0.7, 0.9]), ScriptedData())

    assert trainer.best_epoch == 1
    assert trainer.best_val_loss == pytest.approx(0.3)


def test_best_val_loss_matches_the_history_minimum():
    trainer = Trainer(max_epochs=4, device="cpu", plot=False)
    trainer.fit(ScriptedLoss([0.5, 0.3, 0.7, 0.9]), ScriptedData())

    assert trainer.best_val_loss == pytest.approx(min(trainer.history["val_loss"]))


def test_fit_keeps_the_last_epoch_weights_until_restore_best():
    """fit() 은 가중치를 자동 복원하지 않는다."""
    model = ScriptedLoss([0.5, 0.3, 0.7, 0.9])
    trainer = Trainer(max_epochs=4, device="cpu", plot=False)
    trainer.fit(model, ScriptedData())
    after_fit = {k: v.clone() for k, v in model.state_dict().items()}

    assert trainer.restore_best() == 1

    changed = any(
        not torch.equal(after_fit[k], v) for k, v in model.state_dict().items()
    )
    assert changed


def test_restore_best_before_fit_raises():
    with pytest.raises(RuntimeError, match="has not run"):
        Trainer(max_epochs=1, device="cpu", plot=False).restore_best()


def test_restore_best_without_validation_data_raises():
    trainer = Trainer(max_epochs=2, device="cpu", plot=False)
    trainer.fit(LinReg(), NoValData())

    with pytest.raises(RuntimeError, match="no validation data"):
        trainer.restore_best()


def test_best_is_tracked_even_when_snapshotting_is_off():
    trainer = Trainer(max_epochs=4, device="cpu", plot=False, snapshot_best=False)
    trainer.fit(ScriptedLoss([0.5, 0.3, 0.7, 0.9]), ScriptedData())

    assert trainer.best_epoch == 1
    assert trainer.best_val_loss == pytest.approx(0.3)


def test_restore_best_with_snapshotting_off_reports_the_best_epoch():
    trainer = Trainer(max_epochs=4, device="cpu", plot=False, snapshot_best=False)
    trainer.fit(ScriptedLoss([0.5, 0.3, 0.7, 0.9]), ScriptedData())

    with pytest.raises(RuntimeError, match="epoch 1"):
        trainer.restore_best()


def test_best_path_writes_the_snapshot_to_disk(tmp_path):
    path = tmp_path / "best.pt"
    model = ScriptedLoss([0.5, 0.3, 0.7, 0.9])
    trainer = Trainer(max_epochs=4, device="cpu", plot=False, best_path=path)
    trainer.fit(model, ScriptedData())

    assert path.exists()
    saved = torch.load(path, map_location="cpu", weights_only=False)["model"]

    assert trainer.restore_best() == 1

    for key, value in saved.items():
        assert torch.equal(model.state_dict()[key].cpu(), value)


def test_disk_snapshot_omits_optimizer_state_by_default(tmp_path):
    path = tmp_path / "best.pt"
    trainer = Trainer(max_epochs=3, device="cpu", plot=False, best_path=path)
    trainer.fit(ScriptedLoss([0.5, 0.3, 0.7]), ScriptedData())

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    assert set(ckpt) == {"model", "epoch", "val_loss"}
    assert ckpt["epoch"] == 1
    assert ckpt["val_loss"] == pytest.approx(0.3)


def test_best_with_optim_writes_a_resumable_checkpoint(tmp_path):
    path = tmp_path / "best.pt"
    trainer = Trainer(max_epochs=3, device="cpu", plot=False,
                      best_path=path, best_with_optim=True)
    trainer.fit(ScriptedLoss([0.5, 0.3, 0.7]), ScriptedData())

    fresh = ScriptedLoss([0.0])
    optim = fresh.configure_optimizers()
    meta = Trainer.load_checkpoint(path, fresh, optim)

    assert meta["epoch"] == 1


def test_patience_stops_training_early():
    """0.3 이 최저(epoch 1)이고 patience=2 이므로 epoch 3 에서 멈춘다."""
    trainer = Trainer(max_epochs=5, device="cpu", plot=False, patience=2)
    trainer.fit(ScriptedLoss([0.5, 0.3, 0.7, 0.9, 0.9]), ScriptedData())

    assert len(trainer.history["val_loss"]) == 4   # epoch 0~3
    assert trainer.best_epoch == 1


def test_without_patience_every_epoch_runs():
    trainer = Trainer(max_epochs=5, device="cpu", plot=False)
    trainer.fit(ScriptedLoss([0.5, 0.3, 0.7, 0.9, 0.9]), ScriptedData())

    assert len(trainer.history["val_loss"]) == 5


def test_patience_below_one_is_rejected():
    with pytest.raises(ValueError, match="patience"):
        Trainer(max_epochs=5, device="cpu", plot=False, patience=0)


def test_patience_without_validation_data_is_rejected():
    trainer = Trainer(max_epochs=5, device="cpu", plot=False, patience=2)

    with pytest.raises(ValueError, match="validation data"):
        trainer.fit(LinReg(), NoValData())
