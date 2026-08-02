import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ood.core import add_to_class
from ood.data import DataModule
from ood.evaluate import Predictions, predict
from ood.module import Module
from ood.trainer import Trainer


class ToyClassifierData(DataModule):
    """20개 샘플, 5차원 입력, 3클래스. train 12 / val 8."""

    def __init__(self, batch_size=4):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(0)
        self.X = torch.randn(20, 5)
        self.y = torch.randint(0, 3, (20,))

    def get_dataloader(self, train):
        idx = slice(0, 12) if train else slice(12, None)
        return self.get_tensorloader((self.X, self.y), train, idx)


class ToyClassifier(Module):
    def __init__(self, lr=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Linear(5, 3)


@add_to_class(ToyClassifier)
def loss(self, y_hat, y):
    return F.cross_entropy(y_hat, y)


@add_to_class(ToyClassifier)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)


CPU = torch.device("cpu")


def _predict_val(**kwargs):
    return predict(ToyClassifier(), ToyClassifierData().val_dataloader(), CPU, **kwargs)


def test_predict_collects_every_sample():
    p = _predict_val()

    assert len(p) == 8
    assert p.outputs.shape == (8, 3)
    assert p.targets.shape == (8,)


def test_accuracy_matches_manual_computation():
    p = _predict_val()

    manual = (p.outputs.argmax(dim=-1) == p.targets).float().mean().item()
    assert p.accuracy == manual


def test_inputs_are_not_kept_by_default():
    assert _predict_val().inputs is None


def test_keep_inputs_collects_the_input_tensors():
    p = _predict_val(keep_inputs=True)

    assert p.inputs.shape == (8, 5)


def test_results_are_always_on_cpu():
    """가속기에서 돌려도 결과는 CPU 로 회수돼야 downstream 코드가 받는다."""
    p = _predict_val(keep_inputs=True)

    assert p.outputs.device.type == "cpu"
    assert p.targets.device.type == "cpu"
    assert p.inputs.device.type == "cpu"


def test_derived_properties_are_consistent():
    p = _predict_val()

    assert torch.allclose(p.probs.sum(dim=-1), torch.ones(len(p)))
    expected_conf = p.probs.gather(1, p.preds.unsqueeze(1)).squeeze(1)
    assert torch.allclose(p.confidence, expected_conf)
    assert torch.equal(p.correct, p.preds == p.targets)


def test_predict_leaves_model_in_eval_mode():
    model = ToyClassifier()
    model.train()

    predict(model, ToyClassifierData().val_dataloader(), CPU)

    assert model.training is False


def test_device_defaults_to_the_model_device():
    p = predict(ToyClassifier(), ToyClassifierData().val_dataloader())

    assert p.outputs.device.type == "cpu"


def test_classification_properties_reject_shape_mismatch():
    """회귀 모델의 (N,1) 출력에서 조용히 브로드캐스트되지 않고 막혀야 한다."""
    p = Predictions(torch.randn(4, 1), torch.randn(4, 1))

    with pytest.raises(ValueError, match="분류 전용"):
        p.accuracy


def test_repr_shows_accuracy_for_classification():
    p = Predictions(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]))

    assert "n=4" in repr(p)
    assert "accuracy=" in repr(p)


def test_repr_omits_accuracy_when_shapes_mismatch():
    p = Predictions(torch.randn(4, 1), torch.randn(4, 1))

    assert "accuracy=" not in repr(p)


def _fitted_trainer(data):
    trainer = Trainer(max_epochs=1, device="cpu", plot=False)
    trainer.fit(ToyClassifier(), data)
    return trainer


def test_trainer_predict_uses_the_validation_loader_by_default():
    data = ToyClassifierData()
    trainer = _fitted_trainer(data)

    assert len(trainer.predict(data)) == 8
    assert len(trainer.predict(data, train=True)) == 12


def test_trainer_predict_forwards_keep_inputs():
    data = ToyClassifierData()
    trainer = _fitted_trainer(data)

    p = trainer.predict(data, keep_inputs=True)

    assert p.inputs.shape == (8, 5)
