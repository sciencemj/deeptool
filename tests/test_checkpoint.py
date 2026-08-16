import torch
from torch import nn
import pytest

from deeptool.checkpoint import BestSnapshot


def _model_and_optim():
    model = nn.Linear(1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, optim


def test_best_snapshot_max_mode_returns_whether_score_improved():
    model, optim = _model_and_optim()
    best = BestSnapshot(enabled=False, monitor="iou", mode="max")

    assert best.update(0.4, "epoch", 0, model, optim) is True
    assert best.update(0.4, "epoch", 1, model, optim) is False
    assert best.update(0.3, "epoch", 2, model, optim) is False
    assert best.update(0.6, "epoch", 3, model, optim) is True
    assert best.score == 0.6
    assert best.progress_name == "epoch"
    assert best.progress == 3


def test_default_snapshot_payload_keeps_existing_keys(tmp_path):
    model, optim = _model_and_optim()
    path = tmp_path / "best.pt"
    best = BestSnapshot(path=path)

    best.update(0.3, "epoch", 2, model, optim)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert set(payload) == {"model", "epoch", "val_loss"}
    assert payload["val_loss"] == 0.3


def test_custom_snapshot_payload_names_monitor_and_direction(tmp_path):
    model, optim = _model_and_optim()
    path = tmp_path / "best.pt"
    best = BestSnapshot(path=path, monitor="iou", mode="max")

    best.update(0.7, "epoch", 4, model, optim)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["monitor"] == "iou"
    assert payload["mode"] == "max"
    assert payload["score"] == 0.7
    assert payload["epoch"] == 4


def test_custom_snapshot_with_optimizer_keeps_resume_and_monitor_fields(tmp_path):
    model, optim = _model_and_optim()
    path = tmp_path / "best.pt"
    best = BestSnapshot(
        path=path, with_optim=True, monitor="iou", mode="max"
    )

    best.update(0.8, "epoch", 3, model, optim)
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert {"model", "optim", "epoch", "hparams"} <= payload.keys()
    assert payload["monitor"] == "iou"
    assert payload["mode"] == "max"
    assert payload["score"] == 0.8


def test_disabled_custom_snapshot_error_names_metric_score_and_progress():
    model, optim = _model_and_optim()
    best = BestSnapshot(enabled=False, monitor="iou", mode="max")
    best.update(0.8, "epoch", 3, model, optim)

    with pytest.raises(RuntimeError, match=r"epoch 3.*iou 0.8000"):
        best.restore(model)
