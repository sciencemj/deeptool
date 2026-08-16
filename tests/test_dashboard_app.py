import pytest

pytest.importorskip("streamlit")
pytest.importorskip("altair")

from deeptool._dashboard_app import build_chart
from streamlit.testing.v1 import AppTest


def _write_run(root, name, lines, meta=None):
    run = root / name
    run.mkdir()
    (run / "history.jsonl").write_text("".join(lines))
    if meta is not None:
        import json

        (run / "meta.json").write_text(json.dumps(meta))
    return run


def _app(root):
    source = f'''\
from pathlib import Path
from deeptool._dashboard_app import render_dashboard
render_dashboard(Path({str(root)!r}), refresh=0)
'''
    return AppTest.from_string(source).run(timeout=10)


def _toggles(at):
    return {toggle.label: toggle for toggle in at.sidebar.toggle}


def test_loss_chart_encodes_run_as_color_and_split_as_dash():
    records = [
        {
            "run": "a",
            "unit": "epoch",
            "x": 0,
            "metric": "loss",
            "value": 2.0,
            "split": "train",
        },
        {
            "run": "a",
            "unit": "epoch",
            "x": 0,
            "metric": "loss",
            "value": 2.2,
            "split": "validation",
        },
        {
            "run": "b",
            "unit": "epoch",
            "x": 0,
            "metric": "loss",
            "value": 1.8,
            "split": "train",
        },
    ]

    spec = build_chart(
        records, "loss", "epoch", {"a": "#4f46e5", "b": "#16a34a"}
    ).to_dict()

    assert spec["encoding"]["color"]["field"] == "run"
    assert spec["encoding"]["strokeDash"]["field"] == "split"
    assert spec["encoding"]["strokeDash"]["scale"]["domain"] == [
        "train",
        "validation",
    ]


def test_non_loss_chart_uses_solid_lines_without_split_encoding():
    records = [
        {
            "run": "a",
            "unit": "step",
            "x": 10,
            "metric": "iou",
            "value": 0.5,
            "split": "metric",
        }
    ]

    spec = build_chart(
        records, "iou", "step", {"a": "#4f46e5"}
    ).to_dict()

    assert "strokeDash" not in spec["encoding"]
    assert spec["encoding"]["x"]["title"] == "step"


def test_dashboard_empty_root_has_instructional_state(tmp_path):
    at = _app(tmp_path)

    assert not at.exception
    assert any("No recorded runs" in item.value for item in at.info)


def test_dashboard_exposes_run_toggles_focus_and_loss_chart(tmp_path):
    _write_run(
        tmp_path,
        "exp1",
        ['{"epoch": 0, "train_loss": 2.0, "val_loss": 2.2}\n'],
    )
    _write_run(
        tmp_path, "exp2", ['{"epoch": 0, "train_loss": 1.8}\n']
    )

    at = _app(tmp_path)

    assert not at.exception
    assert set(_toggles(at)) >= {"exp1", "exp2"}
    assert at.sidebar.selectbox[0].label == "Focus run"
    assert len(at.get("vega_lite_chart")) >= 1


def test_hiding_focus_moves_focus_then_all_hidden_shows_prompt(tmp_path):
    _write_run(tmp_path, "exp1", ['{"epoch": 0, "iou": 0.3}\n'])
    _write_run(tmp_path, "exp2", ['{"epoch": 0, "iou": 0.4}\n'])
    at = _app(tmp_path)

    _toggles(at)["exp1"].set_value(False)
    at.run()

    assert at.sidebar.selectbox[0].value == "exp2"

    _toggles(at)["exp2"].set_value(False)
    at.run()

    assert any("Select at least one run" in item.value for item in at.info)


def test_epoch_and_step_runs_render_separate_groups(tmp_path):
    _write_run(tmp_path, "epoch-run", ['{"epoch": 0, "iou": 0.3}\n'])
    _write_run(tmp_path, "step-run", ['{"step": 10, "iou": 0.5}\n'])

    at = _app(tmp_path)

    headings = {item.value for item in at.subheader}
    assert "Epoch metrics" in headings
    assert "Step metrics" in headings


def test_focus_run_populates_progress_latest_row_and_metadata(tmp_path):
    _write_run(
        tmp_path,
        "exp",
        ['{"step": 10, "train_loss": 1.5, "iou": 0.4}\n'],
        meta={"device": "cuda", "model_class": "demo.Model"},
    )

    at = _app(tmp_path)

    assert not at.exception
    assert any(item.label == "Step" and item.value == "10" for item in at.metric)
    assert len(at.json) >= 2


def test_corrupt_run_does_not_hide_healthy_run(tmp_path):
    _write_run(tmp_path, "good", ['{"epoch": 0, "iou": 0.3}\n'])
    _write_run(tmp_path, "bad", ["broken\n"])

    at = _app(tmp_path)

    assert not at.exception
    assert set(_toggles(at)) >= {"good", "bad"}
    assert any("bad" in item.value and "line 1" in item.value for item in at.error)
