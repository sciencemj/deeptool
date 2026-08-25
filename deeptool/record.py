"""Persistent JSONL training records and run comparison plots."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.figure import Figure


RunData = dict[str, dict[str, list[Any]]]


class RunRecorder:
    """Write one run's metadata and completed epoch rows to disk.

    Args:
        log_dir: Directory containing `meta.json` and `history.jsonl`.
    """

    def __init__(self, log_dir: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.log_dir / "meta.json"
        self.history_path = self.log_dir / "history.jsonl"

    def meta(self, **info: object) -> None:
        """Atomically replace this run's metadata document."""
        temporary = self.log_dir / "meta.json.tmp"
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(info, file, ensure_ascii=False, indent=2, default=str)
            file.write("\n")
        temporary.replace(self.meta_path)

    def epoch(self, **scalars: object) -> None:
        """Append and immediately flush one completed epoch row."""
        with self.history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(scalars, ensure_ascii=False) + "\n")
            file.flush()


def _transpose_rows(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {}
    num_rows = len(rows)
    for index, row in enumerate(rows):
        for key, value in row.items():
            column = columns.get(key)
            if column is None:
                column = [None] * num_rows
                columns[key] = column
            column[index] = value
    return columns


def load_runs(root: str | Path) -> RunData:
    """Load immediate child runs below `root` from their JSONL histories.

    Missing metrics are represented by `None` so every metric stays aligned
    with its run's epoch list.

    Args:
        root: Directory whose child directories represent runs.

    Returns:
        Run names mapped to metric names and aligned value lists.

    Raises:
        FileNotFoundError: If `root` does not exist.
        ValueError: If a non-empty history line is not a JSON object.
    """
    runs: RunData = {}
    for run_dir in sorted(Path(root).iterdir()):
        history_path = run_dir / "history.jsonl"
        if not run_dir.is_dir() or not history_path.is_file():
            continue
        rows = _load_history(history_path)
        runs[run_dir.name] = _transpose_rows(rows)
    return runs


def _load_history(
    path: Path, *, allow_incomplete_final: bool = False
) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        line_number = index + 1
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            is_unterminated_final = (
                allow_incomplete_final
                and index == len(lines) - 1
                and not line.endswith(("\n", "\r"))
            )
            if is_unterminated_final:
                break
            raise ValueError(
                f"invalid JSON in {path} at line {line_number}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(
                f"expected a JSON object in {path} at line {line_number}"
            )
        rows.append(row)
    return rows


def plot_runs(source: str | Path | Mapping[str, Mapping[str, list[Any]]]
              ) -> list[Figure]:
    """Plot every recorded metric, overlaying all runs that contain it.

    Args:
        source: A run root accepted by `load_runs`, or its loaded result.

    Returns:
        One Matplotlib figure per metric, excluding `epoch` and `step`.
    """
    runs = source if isinstance(source, Mapping) else load_runs(source)
    metrics = dict.fromkeys(
        metric
        for run in runs.values()
        for metric in run
        if metric not in {"epoch", "step"}
    )
    figures = []
    for metric in metrics:
        figure, axes = plt.subplots()
        xlabels = set()
        for name, run in runs.items():
            if metric not in run:
                continue
            if "epoch" in run:
                xlabel = "epoch"
            elif "step" in run:
                xlabel = "step"
            else:
                xlabel = "index"
            xlabels.add(xlabel)
            x = run.get(xlabel, list(range(len(run[metric]))))
            axes.plot(x, run[metric], label=name)
        axes.set_xlabel(xlabels.pop() if len(xlabels) == 1 else "progress")
        axes.set_ylabel(metric)
        axes.grid(True)
        axes.legend()
        figures.append(figure)
    return figures
