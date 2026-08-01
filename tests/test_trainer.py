import torch
from torch import nn
from torch.nn import functional as F

from ood.core import add_to_class
from ood.data import DataModule
from ood.module import Module
from ood.trainer import Trainer, default_device


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
