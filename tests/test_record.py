import json

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import pytest

from deeptool.record import RunRecorder, _load_history, load_runs, plot_runs


def test_recorder_writes_metadata_and_append_only_epoch_rows(tmp_path):
    recorder = RunRecorder(tmp_path / "exp1")
    recorder.meta(device="cpu", model_class="tests.Model")
    recorder.epoch(epoch=0, train_loss=2.0, iou=0.3)

    first_read = (tmp_path / "exp1" / "history.jsonl").read_text()
    assert json.loads(first_read) == {
        "epoch": 0,
        "train_loss": 2.0,
        "iou": 0.3,
    }

    recorder.epoch(epoch=1, train_loss=1.0, iou=0.5)
    rows = [
        json.loads(line)
        for line in (tmp_path / "exp1" / "history.jsonl").read_text().splitlines()
    ]
    assert rows == [
        {"epoch": 0, "train_loss": 2.0, "iou": 0.3},
        {"epoch": 1, "train_loss": 1.0, "iou": 0.5},
    ]
    assert json.loads((tmp_path / "exp1" / "meta.json").read_text()) == {
        "device": "cpu",
        "model_class": "tests.Model",
    }


def test_load_runs_aligns_sparse_metrics_and_heterogeneous_runs(tmp_path):
    a = RunRecorder(tmp_path / "a")
    a.epoch(epoch=0, iou=0.2)
    a.epoch(epoch=1, acc50=0.7)
    b = RunRecorder(tmp_path / "b")
    b.epoch(epoch=0, ap50=0.4, recall=0.6)

    assert load_runs(tmp_path) == {
        "a": {
            "epoch": [0, 1],
            "iou": [0.2, None],
            "acc50": [None, 0.7],
        },
        "b": {"epoch": [0], "ap50": [0.4], "recall": [0.6]},
    }


def test_load_runs_handles_later_discovered_keys(tmp_path):
    run = RunRecorder(tmp_path / "c")
    run.epoch(epoch=0, loss=1.0)
    run.epoch(epoch=1, loss=0.8, val_loss=0.9)
    run.epoch(epoch=2, loss=0.6)

    assert load_runs(tmp_path) == {
        "c": {
            "epoch": [0, 1, 2],
            "loss": [1.0, 0.8, 0.6],
            "val_loss": [None, 0.9, None],
        }
    }


def test_load_runs_ignores_empty_lines(tmp_path):
    run = tmp_path / "exp"
    run.mkdir()
    (run / "history.jsonl").write_text('\n{"epoch": 0, "iou": 0.2}\n\n')

    assert load_runs(tmp_path) == {"exp": {"epoch": [0], "iou": [0.2]}}


def test_load_runs_reports_malformed_line_location(tmp_path):
    run = tmp_path / "broken"
    run.mkdir()
    (run / "history.jsonl").write_text('{"epoch": 0}\nnot-json\n')

    with pytest.raises(ValueError, match=r"broken/history.jsonl.*line 2"):
        load_runs(tmp_path)


def test_live_reader_ignores_only_non_terminated_partial_final_line(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text('{"epoch": 0, "loss": 1.0}\n{"epoch": 1')

    assert _load_history(path, allow_incomplete_final=True) == [
        {"epoch": 0, "loss": 1.0}
    ]


def test_live_reader_rejects_newline_terminated_invalid_final_line(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text('{"epoch": 0}\nnot-json\n')

    with pytest.raises(ValueError, match=r"history.jsonl.*line 2"):
        _load_history(path, allow_incomplete_final=True)


def test_public_loader_stays_strict_for_partial_final_line(tmp_path):
    run = tmp_path / "exp"
    run.mkdir()
    (run / "history.jsonl").write_text('{"epoch": 0}\n{"epoch": 1')

    with pytest.raises(ValueError, match=r"history.jsonl.*line 2"):
        load_runs(tmp_path)


def test_plot_runs_returns_one_figure_per_metric(tmp_path):
    RunRecorder(tmp_path / "a").epoch(epoch=0, iou=0.2, acc50=0.7)
    RunRecorder(tmp_path / "b").epoch(epoch=0, ap50=0.4)

    figures = plot_runs(tmp_path)

    assert len(figures) == 3
    assert all(isinstance(figure, Figure) for figure in figures)
    assert {figure.axes[0].get_ylabel() for figure in figures} == {
        "iou",
        "acc50",
        "ap50",
    }
    assert all(figure.number in plt.get_fignums() for figure in figures)
    plt.close("all")


def test_plot_runs_uses_step_axis_for_step_records(tmp_path):
    recorder = RunRecorder(tmp_path / "llm")
    recorder.epoch(step=10, train_loss=2.0)
    recorder.epoch(step=20, train_loss=1.5)

    figure = plot_runs(tmp_path)[0]
    axes = figure.axes[0]

    assert axes.get_xlabel() == "step"
    assert list(axes.lines[0].get_xdata()) == [10, 20]
    plt.close("all")
