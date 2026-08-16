"""Streamlit-independent loading and normalization for the run dashboard."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal

from deeptool.record import _load_history


@dataclass(frozen=True)
class DashboardRun:
    """One discovered run, including isolated history or metadata errors."""

    name: str
    directory: Path
    metrics: dict[str, list[Any]]
    meta: dict[str, Any]
    modified_at: float
    unit: Literal["epoch", "step"] | None
    progress: int | float | None
    error: str | None = None
    meta_error: str | None = None


def discover_runs(root: str | Path) -> list[DashboardRun]:
    """Discover immediate run directories without one bad run blocking others."""
    runs = []
    for run_dir in sorted(Path(root).iterdir()):
        history_path = run_dir / "history.jsonl"
        if not run_dir.is_dir() or not history_path.is_file():
            continue
        metrics: dict[str, list[Any]] = {}
        error = None
        try:
            rows = _load_history(
                history_path, allow_incomplete_final=True
            )
            keys = dict.fromkeys(key for row in rows for key in row)
            metrics = {
                key: [row.get(key) for row in rows]
                for key in keys
            }
        except (OSError, ValueError) as caught:
            error = str(caught)

        meta, meta_error = _load_meta(run_dir / "meta.json")
        unit = _progress_unit(metrics)
        progress = _last_present(metrics.get(unit, [])) if unit else None
        try:
            modified_at = history_path.stat().st_mtime
        except OSError:
            modified_at = 0.0
        runs.append(DashboardRun(
            name=run_dir.name,
            directory=run_dir,
            metrics=metrics,
            meta=meta,
            modified_at=modified_at,
            unit=unit,
            progress=progress,
            error=error,
            meta_error=meta_error,
        ))
    return runs


def metric_groups(runs: list[DashboardRun]) -> list[str]:
    """Return the combined loss group followed by sorted custom metrics."""
    names = {
        metric
        for run in runs
        if run.error is None
        for metric in run.metrics
        if metric not in {"epoch", "step", "train_loss", "val_loss"}
    }
    losses_present = any(
        {"train_loss", "val_loss"} & run.metrics.keys()
        for run in runs
        if run.error is None
    )
    return (["loss"] if losses_present else []) + sorted(names)


def chart_records(
    runs: list[DashboardRun],
    metric: str,
    unit: Literal["epoch", "step"],
) -> list[dict[str, object]]:
    """Normalize scalar values into long-form chart records."""
    records = []
    source_metrics = (
        (("train_loss", "train"), ("val_loss", "validation"))
        if metric == "loss"
        else ((metric, "metric"),)
    )
    for run in runs:
        if run.error is not None or run.unit != unit:
            continue
        progress = run.metrics.get(unit, [])
        for source_metric, split in source_metrics:
            values = run.metrics.get(source_metric, [])
            for x, value in zip(progress, values, strict=False):
                if not _finite_number(x) or not _finite_number(value):
                    continue
                records.append({
                    "run": run.name,
                    "unit": unit,
                    "x": x,
                    "metric": metric,
                    "value": value,
                    "split": split,
                })
    return records


def _load_meta(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object in {path}")
        return value, None
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {}, f"{path}: {error}"


def _progress_unit(
    metrics: dict[str, list[Any]],
) -> Literal["epoch", "step"] | None:
    if "epoch" in metrics:
        return "epoch"
    if "step" in metrics:
        return "step"
    return None


def _last_present(values: list[Any]) -> int | float | None:
    return next((value for value in reversed(values) if value is not None), None)


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )
