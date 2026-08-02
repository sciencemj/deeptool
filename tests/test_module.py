import pytest
import torch
from torch import nn
from torch.nn import functional as F

from deeptool.board import ProgressBoard
from deeptool.core import add_to_class
from deeptool.module import Module


class FakeTrainer:
    train_batch_idx = 4
    num_train_batches = 8
    num_val_batches = 2
    epoch = 0


class ToyNet(Module):
    def __init__(self, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Linear(3, 1)


@add_to_class(ToyNet)
def loss(self, y_hat, y):
    return F.mse_loss(y_hat, y)


def _batch():
    return torch.zeros(5, 3), torch.zeros(5, 1)


def test_forward_delegates_to_self_net():
    model = ToyNet()
    out = model(torch.zeros(5, 3))

    assert out.shape == (5, 1)


def test_forward_without_net_raises():
    class NoNet(Module):
        pass

    with pytest.raises(AssertionError):
        NoNet()(torch.zeros(1, 3))


def test_loss_is_abstract():
    class Bare(Module):
        pass

    with pytest.raises(NotImplementedError):
        Bare().loss(None, None)


def test_configure_optimizers_is_abstract():
    with pytest.raises(NotImplementedError):
        ToyNet().configure_optimizers()


def test_training_step_returns_a_scalar_tensor():
    out = ToyNet().training_step(_batch())

    assert torch.is_tensor(out)
    assert out.ndim == 0


def test_validation_step_returns_a_scalar_tensor():
    out = ToyNet().validation_step(_batch())

    assert torch.is_tensor(out)
    assert out.ndim == 0


def test_plot_is_a_noop_without_a_board():
    ToyNet().plot("loss", torch.tensor(1.0), train=True)  # 예외 없이 통과


def test_plot_writes_to_the_board_with_a_train_prefix():
    model = ToyNet()
    model.board = ProgressBoard(display=False)
    model.trainer = FakeTrainer()

    model.plot("loss", torch.tensor(2.0), train=True)

    # every_n = num_train_batches / plot_train_per_epoch = 8 / 2 = 4 → 아직 버퍼
    assert model.board.data["train_loss"] == []
    assert model.board.raw_points["train_loss"] == [(0.5, 2.0)]


def test_plot_uses_epoch_as_x_for_validation():
    model = ToyNet()
    model.board = ProgressBoard(display=False)
    model.trainer = FakeTrainer()

    # every_n = num_val_batches / plot_valid_per_epoch = 2 / 1 = 2
    model.plot("loss", torch.tensor(2.0), train=False)
    model.plot("loss", torch.tensor(4.0), train=False)

    assert model.board.data["val_loss"] == [(1.0, 3.0)]


def test_plot_accepts_plain_floats():
    model = ToyNet()
    model.board = ProgressBoard(display=False)
    model.trainer = FakeTrainer()

    model.plot("acc", 0.75, train=True)

    assert model.board.raw_points["train_acc"] == [(0.5, 0.75)]
