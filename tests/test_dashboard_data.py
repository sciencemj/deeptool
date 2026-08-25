import json

from deeptool._dashboard_data import (
    chart_records,
    discover_runs,
    metric_groups,
)


def _write_run(root, name, lines, meta=None):
    run = root / name
    run.mkdir()
    (run / "history.jsonl").write_text("".join(lines))
    if meta is not None:
        (run / "meta.json").write_text(json.dumps(meta))
    return run


def test_discover_runs_keeps_healthy_run_when_another_is_corrupt(tmp_path):
    _write_run(tmp_path, "good", ['{"epoch": 0, "iou": 0.4}\n'])
    _write_run(tmp_path, "bad", ['{"epoch": 0}\n', "broken\n"])

    runs = {run.name: run for run in discover_runs(tmp_path)}

    assert runs["good"].error is None
    assert runs["good"].metrics["iou"] == [0.4]
    assert "line 2" in runs["bad"].error


def test_discover_runs_uses_history_without_metadata(tmp_path):
    _write_run(tmp_path, "exp", ['{"step": 10, "loss": 2.0}\n'])

    run = discover_runs(tmp_path)[0]

    assert run.meta == {}
    assert run.unit == "step"
    assert run.progress == 10


def test_invalid_metadata_does_not_block_history(tmp_path):
    run_dir = _write_run(
        tmp_path, "exp", ['{"epoch": 0, "loss": 2.0}\n']
    )
    (run_dir / "meta.json").write_text("{")

    run = discover_runs(tmp_path)[0]

    assert run.error is None
    assert run.meta == {}
    assert "meta.json" in run.meta_error


def test_loss_group_uses_split_labels_and_arbitrary_metrics_stay_separate(
    tmp_path,
):
    _write_run(
        tmp_path,
        "a",
        ['{"epoch": 0, "train_loss": 2.0, "val_loss": 2.2, "iou": 0.3}\n'],
    )
    _write_run(
        tmp_path,
        "b",
        ['{"epoch": 0, "train_loss": 1.8, "ap50": 0.4}\n'],
    )
    runs = discover_runs(tmp_path)

    assert metric_groups(runs) == ["loss", "ap50", "iou"]
    loss = chart_records(runs, "loss", "epoch")
    assert {(row["run"], row["split"]) for row in loss} == {
        ("a", "train"),
        ("a", "validation"),
        ("b", "train"),
    }


def test_chart_records_never_mix_epoch_and_step_runs(tmp_path):
    _write_run(tmp_path, "epoch-run", ['{"epoch": 0, "iou": 0.3}\n'])
    _write_run(tmp_path, "step-run", ['{"step": 5, "iou": 0.5}\n'])
    runs = discover_runs(tmp_path)

    assert {row["run"] for row in chart_records(runs, "iou", "epoch")} == {
        "epoch-run"
    }
    assert {row["run"] for row in chart_records(runs, "iou", "step")} == {
        "step-run"
    }


def test_chart_records_skip_non_finite_and_non_numeric_values(tmp_path):
    _write_run(
        tmp_path,
        "exp",
        [
            '{"epoch": 0, "score": 0.4}\n',
            '{"epoch": 1, "score": null}\n',
            '{"epoch": 2, "score": true}\n',
            '{"epoch": 3, "score": "bad"}\n',
        ],
    )

    records = chart_records(discover_runs(tmp_path), "score", "epoch")

    assert records == [
        {
            "run": "exp",
            "unit": "epoch",
            "x": 0,
            "metric": "score",
            "value": 0.4,
            "split": "metric",
        }
    ]


def test_discover_runs_transposes_sparse_keys_correctly(tmp_path):
    _write_run(
        tmp_path,
        "sparse",
        [
            '{"epoch": 0, "metric_a": 1}\n',
            '{"epoch": 1, "metric_b": 2}\n',
            '{"epoch": 2, "metric_a": 3, "metric_b": 4}\n',
        ],
    )

    run = discover_runs(tmp_path)[0]

    assert run.metrics["epoch"] == [0, 1, 2]
    assert run.metrics["metric_a"] == [1, None, 3]
    assert run.metrics["metric_b"] == [None, 2, 4]
