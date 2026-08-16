# Streamlit Run Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an optional, live-refreshing Streamlit dashboard that compares arbitrary epoch/step runs locally or through a safe SSH tunnel, with per-run visibility and Focus controls.

**Architecture:** Reuse the strict JSONL reader through a private live-read option, normalize each run in a Streamlit-independent data module, and launch a packaged Streamlit script through its public CLI. Keep all Streamlit/Altair imports in optional dashboard modules so core `import deeptool` has no new dependency.

**Tech Stack:** Python 3.11+, standard-library JSON/argparse/subprocess, Streamlit 1.37+, Altair 5+, pytest 8+, Streamlit `AppTest`

## Global Constraints

- Complete the configurable-monitor and step-training plans first.
- Core dependencies remain exactly Torch, Matplotlib, and IPython.
- `streamlit>=1.37` and `altair>=5` belong only to `deeptool[dashboard]`.
- The command is `deeptool-dashboard RUNS_DIR`; default bind is `127.0.0.1:8501`.
- Default refresh is 2 seconds; refresh `0` disables automatic refresh.
- Live reading may ignore only an incomplete, non-newline-terminated final JSONL fragment.
- A corrupt run must not prevent healthy runs from rendering.
- `train_loss` and `val_loss` share one chart: run is color, train is solid, validation is dashed.
- Arbitrary non-loss scalar names remain uninterpreted and receive independent charts.
- Epoch and step runs must never share one x axis.
- Visibility toggles, Focus run, metric toggles, stable run colors, and Focus-based KPI/latest/meta panels must match the approved mockup.
- The dashboard is read-only and adds no authentication, TLS, training control, or custom remote protocol.
- Release all three approved features together as deeptool `0.4.0` only after full test, docs, wheel, and dashboard smoke gates pass.

---

## File Structure

- Modify `deeptool/record.py`: add a private live-safe final-fragment option while preserving strict `load_runs()`.
- Create `deeptool/_dashboard_data.py`: discover runs, isolate errors, load optional metadata, identify progress units, and create scalar chart records.
- Create `deeptool/dashboard.py`: dependency check, CLI parsing, path validation, and Streamlit subprocess launch.
- Create `deeptool/_dashboard_app.py`: Streamlit state, refresh fragment, approved layout, and Altair charts.
- Modify `pyproject.toml` and `uv.lock`: optional extra and console entry point.
- Create `tests/test_dashboard_data.py`, `tests/test_dashboard_cli.py`, and `tests/test_dashboard_app.py`: pure data, launcher, chart, and UI tests.
- Modify `.github/workflows/ci.yml`: retain core matrix and add one dashboard-extra job.
- Create `docs/guide/dashboard.md` and `docs/guide/dashboard.en.md`; update navigation, README, and API-adjacent guidance.
- Modify `deeptool/__init__.py` and `tests/test_package.py`: `0.4.0` release version.

### Task 1: Make the Private JSONL Reader Safe for Concurrent Final Writes

**Files:**
- Modify: `deeptool/record.py`
- Modify: `tests/test_record.py`

**Interfaces:**
- Consumes: existing `_load_history(path)` and strict `load_runs(root)`.
- Produces: `_load_history(path, *, allow_incomplete_final=False) -> list[dict[str, Any]]`; unchanged public strict behavior.

- [ ] **Step 1: Write failing strict-versus-live final-fragment tests**

```python
# tests/test_record.py
from pathlib import Path

from deeptool.record import _load_history


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
```

- [ ] **Step 2: Run reader tests and confirm the private keyword fails**

Run: `rtk uv run pytest tests/test_record.py -k 'live_reader or public_loader_stays' -q`

Expected: FAIL because `_load_history()` has no `allow_incomplete_final` argument.

- [ ] **Step 3: Parse with line endings retained and skip only the allowed fragment**

```python
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
```

Do not pass the flag from public `load_runs()`.

- [ ] **Step 4: Run all record tests**

Run: `rtk uv run pytest tests/test_record.py -q`

Expected: PASS, including existing blank-line, heterogeneous-schema, and malformed-middle-line behavior.

- [ ] **Step 5: Commit the live-safe reader**

```bash
rtk git add deeptool/record.py tests/test_record.py
rtk git commit -m "feat: tolerate partial live history fragments"
```

### Task 2: Build a Streamlit-Independent Dashboard Data Model

**Files:**
- Create: `deeptool/_dashboard_data.py`
- Create: `tests/test_dashboard_data.py`

**Interfaces:**
- Consumes: `_load_history(path, allow_incomplete_final=True)` and immediate run directories.
- Produces: `DashboardRun`; `discover_runs(root) -> list[DashboardRun]`; `metric_groups(runs) -> list[str]`; `chart_records(runs, metric, unit) -> list[dict[str, object]]`.

- [ ] **Step 1: Write failing discovery and error-isolation tests**

```python
# tests/test_dashboard_data.py
import json

from deeptool._dashboard_data import discover_runs


def _write_run(root, name, lines, meta=None):
    run = root / name
    run.mkdir()
    (run / "history.jsonl").write_text("".join(lines))
    if meta is not None:
        (run / "meta.json").write_text(json.dumps(meta))
    return run


def test_discover_runs_keeps_healthy_run_when_another_is_corrupt(tmp_path):
    _write_run(tmp_path, "good", ['{"epoch": 0, "iou": 0.4}\n'])
    _write_run(tmp_path, "bad", ['{"epoch": 0}\n', 'broken\n'])

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
```

- [ ] **Step 2: Run discovery tests and confirm the module is missing**

Run: `rtk uv run pytest tests/test_dashboard_data.py -q`

Expected: FAIL with `ModuleNotFoundError: deeptool._dashboard_data`.

- [ ] **Step 3: Implement the immutable run model and isolated discovery**

```python
# deeptool/_dashboard_data.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from deeptool.record import _load_history


@dataclass(frozen=True)
class DashboardRun:
    name: str
    directory: Path
    metrics: dict[str, list[Any]]
    meta: dict[str, Any]
    modified_at: float
    unit: Literal["epoch", "step"] | None
    progress: int | float | None
    error: str | None = None
    meta_error: str | None = None
```

`discover_runs()` sorts immediate child directories, ignores children without
`history.jsonl`, calls the live reader inside a per-run `try`, aligns row keys
with `None` exactly like `load_runs()`, and loads metadata in a second isolated
`try`. A history failure returns a `DashboardRun` with empty metrics and its
error text. Unit is `epoch` when that key exists, otherwise `step`; progress is
the last non-`None` structural value.

- [ ] **Step 4: Write failing metric grouping and chart-record tests**

```python
from deeptool._dashboard_data import chart_records, metric_groups


def test_loss_group_uses_split_labels_and_arbitrary_metrics_stay_separate(tmp_path):
    _write_run(
        tmp_path, "a",
        ['{"epoch": 0, "train_loss": 2.0, "val_loss": 2.2, "iou": 0.3}\n'],
    )
    _write_run(
        tmp_path, "b",
        ['{"epoch": 0, "train_loss": 1.8, "ap50": 0.4}\n'],
    )
    runs = discover_runs(tmp_path)

    assert metric_groups(runs) == ["loss", "ap50", "iou"]
    loss = chart_records(runs, "loss", "epoch")
    assert {(row["run"], row["split"]) for row in loss} == {
        ("a", "train"), ("a", "validation"), ("b", "train")
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
```

- [ ] **Step 5: Implement deterministic scalar normalization**

Exclude `epoch` and `step` from metric groups. Replace the exact keys
`train_loss` and `val_loss` with one leading `loss` group. Sort all other
metric names. `chart_records()` emits long-form records with `run`, `unit`,
`x`, `metric`, `value`, and `split`; loss split is `train` or `validation`, and
other metrics use `metric`. Skip `None`, bool, non-numeric, NaN, and infinity
without failing the run.

- [ ] **Step 6: Run pure dashboard data tests**

Run: `rtk uv run pytest tests/test_dashboard_data.py tests/test_record.py -q`

Expected: PASS without Streamlit or Altair installed.

- [ ] **Step 7: Commit the data model**

```bash
rtk git add deeptool/_dashboard_data.py tests/test_dashboard_data.py
rtk git commit -m "feat: normalize dashboard run data"
```

### Task 3: Package the Optional Extra and Safe CLI Launcher

**Files:**
- Create: `deeptool/dashboard.py`
- Create: `tests/test_dashboard_cli.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: an existing run-root directory and the installed Streamlit module.
- Produces: `deeptool-dashboard`; `dashboard.main(argv: Sequence[str] | None = None) -> int`; Streamlit app arguments `RUNS_DIR --refresh SECONDS`.

- [ ] **Step 1: Write failing parser and subprocess-forwarding tests**

```python
# tests/test_dashboard_cli.py
import sys

import pytest

from deeptool import dashboard


def test_dashboard_cli_uses_safe_defaults(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(dashboard.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        dashboard.subprocess,
        "run",
        lambda command, check: captured.update(command=command, check=check),
    )

    assert dashboard.main([str(tmp_path)]) == 0
    command = captured["command"]
    assert command[:3] == [sys.executable, "-m", "streamlit"]
    assert "--server.address=127.0.0.1" in command
    assert "--server.port=8501" in command
    assert command[-3:] == [str(tmp_path.resolve()), "--refresh", "2.0"]
    assert captured["check"] is True


def test_dashboard_cli_forwards_remote_options(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(dashboard.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        dashboard.subprocess,
        "run",
        lambda command, check: captured.update(command=command),
    )

    dashboard.main([
        str(tmp_path), "--host", "0.0.0.0", "--port", "9000",
        "--refresh", "5", "--no-browser",
    ])

    assert "--server.address=0.0.0.0" in captured["command"]
    assert "--server.port=9000" in captured["command"]
    assert "--server.headless=true" in captured["command"]


def test_dashboard_cli_reports_missing_extra(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dashboard.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(SystemExit) as error:
        dashboard.main([str(tmp_path)])

    assert error.value.code == 2
    assert 'uv add "deeptool[dashboard]"' in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ["--host", ""],
        ["--port", "0"],
        ["--port", "65536"],
        ["--refresh", "-1"],
        ["--refresh", "nan"],
        ["--refresh", "inf"],
    ],
)
def test_dashboard_cli_rejects_invalid_network_and_refresh_options(
    tmp_path, arguments
):
    with pytest.raises(SystemExit) as error:
        dashboard.main([str(tmp_path), *arguments])
    assert error.value.code == 2


def test_dashboard_cli_rejects_missing_and_file_roots(tmp_path):
    file_root = tmp_path / "history.jsonl"
    file_root.write_text("")

    for root in (tmp_path / "missing", file_root):
        with pytest.raises(SystemExit) as error:
            dashboard.main([str(root)])
        assert error.value.code == 2
```

- [ ] **Step 2: Run CLI tests and confirm the module is missing**

Run: `rtk uv run pytest tests/test_dashboard_cli.py -q`

Expected: FAIL with `ImportError` because `deeptool.dashboard` does not exist.

- [ ] **Step 3: Implement argument validation and the public Streamlit command**

```python
# deeptool/dashboard.py
import argparse
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deeptool-dashboard")
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--refresh", type=float, default=2.0)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    root = args.runs_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"runs directory does not exist: {root}")
    if not args.host:
        parser.error("host must be non-empty")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if not math.isfinite(args.refresh) or args.refresh < 0:
        parser.error("refresh must be a finite non-negative number")
    if importlib.util.find_spec("streamlit") is None:
        parser.error('dashboard dependencies are missing; run: uv add "deeptool[dashboard]"')
    app = Path(__file__).with_name("_dashboard_app.py")
    command = [
        sys.executable, "-m", "streamlit", "run", str(app),
        f"--server.address={args.host}", f"--server.port={args.port}",
        f"--server.headless={str(args.no_browser).lower()}",
        "--", str(root), "--refresh", str(args.refresh),
    ]
    subprocess.run(command, check=True)
    return 0
```

The CLI test mocks process launch and therefore does not import the app file;
Task 4 creates the packaged `_dashboard_app.py` before any end-to-end launch.

- [ ] **Step 4: Declare the extra and entry point, then update the lockfile**

```toml
[project.optional-dependencies]
dashboard = [
    "streamlit>=1.37",
    "altair>=5",
]

[project.scripts]
deeptool-dashboard = "deeptool.dashboard:main"
```

Run: `rtk uv lock`

Expected: `uv.lock` includes Streamlit and Altair only through the `dashboard` optional dependency metadata.

- [ ] **Step 5: Run CLI, package, and core-import tests**

Run: `rtk uv run --extra dashboard pytest tests/test_dashboard_cli.py tests/test_package.py -q`

Expected: PASS.

Run: `rtk uv run python -c "import deeptool; print(deeptool.__version__)"`

Expected: prints the current version without importing Streamlit from `deeptool.__init__`.

- [ ] **Step 6: Commit optional packaging and launcher**

```bash
rtk git add pyproject.toml uv.lock deeptool/dashboard.py tests/test_dashboard_cli.py
rtk git commit -m "feat: add optional dashboard launcher"
```

### Task 4: Render the Approved Streamlit Dashboard

**Files:**
- Create: `deeptool/_dashboard_app.py`
- Create: `tests/test_dashboard_app.py`

**Interfaces:**
- Consumes: `discover_runs()`, `metric_groups()`, `chart_records()`, command-supplied root and refresh interval.
- Produces: `render_dashboard(root: Path, refresh: float) -> None`; `build_chart(records, metric, unit, colors) -> alt.Chart`; executable app `main(argv=None)`.

- [ ] **Step 1: Write failing Altair encoding tests**

```python
# tests/test_dashboard_app.py
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("altair")

from deeptool._dashboard_app import build_chart


def test_loss_chart_encodes_run_as_color_and_split_as_dash():
    records = [
        {"run": "a", "unit": "epoch", "x": 0, "metric": "loss", "value": 2.0, "split": "train"},
        {"run": "a", "unit": "epoch", "x": 0, "metric": "loss", "value": 2.2, "split": "validation"},
        {"run": "b", "unit": "epoch", "x": 0, "metric": "loss", "value": 1.8, "split": "train"},
    ]
    spec = build_chart(
        records, "loss", "epoch", {"a": "#4f46e5", "b": "#16a34a"}
    ).to_dict()

    assert spec["encoding"]["color"]["field"] == "run"
    assert spec["encoding"]["strokeDash"]["field"] == "split"
    assert spec["encoding"]["strokeDash"]["scale"]["domain"] == [
        "train", "validation"
    ]


def test_non_loss_chart_uses_solid_lines_without_split_encoding():
    records = [
        {"run": "a", "unit": "step", "x": 10, "metric": "iou", "value": 0.5, "split": "metric"}
    ]
    spec = build_chart(records, "iou", "step", {"a": "#4f46e5"}).to_dict()

    assert "strokeDash" not in spec["encoding"]
    assert spec["encoding"]["x"]["title"] == "step"
```

- [ ] **Step 2: Run chart tests and confirm the app module is missing**

Run: `rtk uv run --extra dashboard pytest tests/test_dashboard_app.py -q`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement long-form Altair charts**

```python
# deeptool/_dashboard_app.py
def build_chart(records, metric, unit, colors):
    data = alt.Data(values=records)
    encoding = {
        "x": alt.X("x:Q", title=unit),
        "y": alt.Y("value:Q", title="Loss" if metric == "loss" else metric),
        "color": alt.Color(
            "run:N",
            scale=alt.Scale(domain=list(colors), range=list(colors.values())),
        ),
        "tooltip": ["run:N", "x:Q", "value:Q"],
    }
    if metric == "loss":
        encoding["strokeDash"] = alt.StrokeDash(
            "split:N",
            scale=alt.Scale(
                domain=["train", "validation"],
                range=[[1, 0], [6, 4]],
            ),
        )
    return alt.Chart(data).mark_line().encode(**encoding).interactive()
```

- [ ] **Step 4: Write failing Streamlit AppTest state and empty-view tests**

```python
from streamlit.testing.v1 import AppTest


def _write_run(root, name, lines):
    run = root / name
    run.mkdir()
    (run / "history.jsonl").write_text("".join(lines))
    return run


def _app(root):
    source = f'''\
from pathlib import Path
from deeptool._dashboard_app import render_dashboard
render_dashboard(Path({str(root)!r}), refresh=0)
'''
    return AppTest.from_string(source).run()


def test_dashboard_empty_root_has_instructional_state(tmp_path):
    at = _app(tmp_path)
    assert not at.exception
    assert any("No recorded runs" in item.value for item in at.info)


def test_dashboard_exposes_run_toggles_focus_and_loss_chart(tmp_path):
    _write_run(tmp_path, "exp1", ['{"epoch": 0, "train_loss": 2.0, "val_loss": 2.2}\n'])
    _write_run(tmp_path, "exp2", ['{"epoch": 0, "train_loss": 1.8}\n'])

    at = _app(tmp_path)

    assert {toggle.label for toggle in at.sidebar.toggle} >= {"exp1", "exp2"}
    assert at.sidebar.selectbox[0].label == "Focus run"
    assert len(at.get("vega_lite_chart")) >= 1
```

- [ ] **Step 5: Implement session state, sidebar, Focus panels, and refresh fragment**

`render_dashboard()` must:

1. Call `st.set_page_config(page_title="deeptool dashboard", layout="wide")`.
2. Initialize `visible_runs`, `run_colors`, and `focus_run` in
   `st.session_state`; assign new colors from a fixed accessible palette
   without changing existing mappings.
3. Render one sidebar toggle per healthy or errored run, a Focus selectbox over
   visible healthy runs, metric toggles, refresh status, and resolved root.
4. On a hidden Focus run, choose the first remaining healthy visible run.
5. Show per-run errors independently and the all-hidden selection prompt.
6. Render Focus KPI cards for progress, train loss, validation loss, and the
   first selected custom metric; then latest row and metadata.
7. For each present unit (`epoch`, `step`), render a separate heading and chart
   group. Use one combined loss chart and one chart per selected arbitrary
   metric.
8. Show last history modification time, not an inferred LIVE/done state.

Wrap only the data-dependent body in a fragment:

```python
run_every = refresh if refresh > 0 else None

@st.fragment(run_every=run_every)
def live_panel():
    runs = discover_runs(root)
    # render controls and charts; widget keys keep state across reruns

live_panel()
```

`main(argv=None)` parses exactly `RUNS_DIR --refresh SECONDS` from the arguments
placed after Streamlit's `--` separator and calls `render_dashboard()`.

- [ ] **Step 6: Complete AppTest interactions**

Add the exact interaction and unit-separation tests:

```python
def _toggles(at):
    return {toggle.label: toggle for toggle in at.sidebar.toggle}


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
```

Use deterministic widget keys derived from run names so reruns preserve each
toggle and AppTest can mutate them reliably.

- [ ] **Step 7: Run dashboard app and data tests**

Run: `rtk uv run --extra dashboard pytest tests/test_dashboard_app.py tests/test_dashboard_data.py tests/test_dashboard_cli.py -q`

Expected: PASS with no AppTest exceptions.

- [ ] **Step 8: Commit the approved dashboard UI**

```bash
rtk git add deeptool/_dashboard_app.py tests/test_dashboard_app.py
rtk git commit -m "feat: render live run dashboard"
```

### Task 5: Document Local and SSH Dashboard Workflows

**Files:**
- Create: `docs/guide/dashboard.md`
- Create: `docs/guide/dashboard.en.md`
- Modify: `mkdocs.yml`
- Modify: `README.md`
- Modify: `docs/index.md`
- Modify: `docs/index.en.md`
- Modify: `docs/api.md`

**Interfaces:**
- Consumes: finalized CLI and UI behavior.
- Produces: Korean/English install, local, remote, chart, and security guidance.

- [ ] **Step 1: Add the canonical local workflow in both languages**

```text
uv add "deeptool[dashboard]"
deeptool-dashboard runs/
```

Explain automatic refresh, refresh `0`, per-run toggles, Focus run semantics,
combined solid/dashed loss, arbitrary metrics, and separate epoch/step groups.

- [ ] **Step 2: Add the safe SSH workflow and exposure warning**

```text
# local machine
ssh -L 8501:127.0.0.1:8501 user@training-host

# remote training host
deeptool-dashboard runs/ --no-browser
```

Tell the user to open `http://127.0.0.1:8501`. Document
`--host 0.0.0.0 --port 8501` separately with a warning that deeptool supplies
no authentication or TLS and that reachable networks can access the dashboard.

- [ ] **Step 3: Add the guide to navigation and discovery pages**

Add `guide/dashboard.md` beneath Trainer in `mkdocs.yml`, with English nav
translation `Run dashboard`. Link it from both index pages and README. Add the
public `deeptool.dashboard.main` launcher to `docs/api.md` without exporting it
from `deeptool.__all__`.

- [ ] **Step 4: Build strict bilingual documentation**

Run: `rtk uv run --group docs mkdocs build --strict`

Expected: PASS with both `dashboard.md` language variants and no missing nav translation.

- [ ] **Step 5: Commit dashboard documentation**

```bash
rtk git add README.md mkdocs.yml docs/guide/dashboard.md docs/guide/dashboard.en.md docs/index.md docs/index.en.md docs/api.md
rtk git commit -m "docs: add local and SSH dashboard guide"
```

### Task 6: Add Dashboard CI and Wheel Smoke Coverage

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_package.py`

**Interfaces:**
- Consumes: optional dependency metadata, console entry point, full test suite.
- Produces: core matrix without dashboard installation plus one Python 3.13 dashboard job.

- [ ] **Step 1: Add package metadata assertions for the optional extra and script**

```python
# tests/test_package.py
from importlib.metadata import entry_points, requires


def test_dashboard_console_script_is_installed():
    scripts = {item.name: item.value for item in entry_points(group="console_scripts")}
    assert scripts["deeptool-dashboard"] == "deeptool.dashboard:main"


def test_dashboard_dependencies_are_optional():
    requirements = requires("deeptool") or []
    assert any(
        item.startswith("streamlit>=1.37") and "dashboard" in item
        for item in requirements
    )
    assert any(
        item.startswith("altair>=5") and "dashboard" in item
        for item in requirements
    )
```

- [ ] **Step 2: Keep core CI unchanged and add a dashboard job**

```yaml
  dashboard:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - name: Install uv
        uses: astral-sh/setup-uv@v9.0.0
      - name: Install dashboard extra
        run: uv sync --locked --python 3.13 --extra dashboard
      - name: Run dashboard tests
        run: >-
          uv run pytest -v
          tests/test_dashboard_data.py
          tests/test_dashboard_cli.py
          tests/test_dashboard_app.py
          tests/test_package.py
      - name: Build wheel
        run: uv build
      - name: Smoke console script from built wheel
        run: |
          WHEEL="$(find dist -name 'deeptool-*.whl' -print -quit)"
          uv run --no-project --isolated --with "$WHEEL[dashboard]" \
            deeptool-dashboard --help
```

Do not add `--extra dashboard` to the existing 3.11/3.12/3.13 core matrix; that
matrix proves optional imports stay optional.

- [ ] **Step 3: Run local core and dashboard gates**

Run: `rtk uv run pytest -q`

Expected: PASS without explicitly selecting the dashboard extra.

Run: `rtk uv run --extra dashboard pytest tests/test_dashboard_data.py tests/test_dashboard_cli.py tests/test_dashboard_app.py tests/test_package.py -q`

Expected: PASS.

Run: `rtk uv build`

Expected: sdist and wheel build successfully and the wheel contains all four dashboard Python modules plus console metadata.

- [ ] **Step 4: Commit CI and metadata coverage**

```bash
rtk git add .github/workflows/ci.yml tests/test_package.py
rtk git commit -m "ci: verify optional dashboard package"
```

### Task 7: Integrate, Version, Push, and Publish deeptool 0.4.0

**Files:**
- Modify: `deeptool/__init__.py`
- Modify: `tests/test_package.py`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: completed monitor, step-training, and dashboard plans with clean per-task commits.
- Produces: GitHub `v0.4.0` release and PyPI `deeptool==0.4.0` through the existing trusted-publishing workflow.

- [ ] **Step 1: Confirm release preconditions without changing state**

Run: `rtk git status --short --branch`

Expected: clean `main`, ahead only by the reviewed feature commits.

Run: `rtk git log --oneline origin/main..HEAD`

Expected: only the approved design, implementation, documentation, test, and release-preparation commits.

Check PyPI and GitHub releases to confirm `0.4.0` does not already exist before
changing the version.

- [ ] **Step 2: Bump the package and locked project version**

Change:

```python
# deeptool/__init__.py
__version__ = "0.4.0"
```

Update `tests/test_package.py` to expect `0.4.0`, then run `rtk uv lock` so the local
project entry in `uv.lock` matches.

- [ ] **Step 3: Run the complete release gates**

Run: `rtk uv run pytest -q`

Expected: PASS.

Run: `rtk uv run --extra dashboard pytest tests/test_dashboard_data.py tests/test_dashboard_cli.py tests/test_dashboard_app.py tests/test_package.py -q`

Expected: PASS.

Run: `rtk uv run --group docs mkdocs build --strict`

Expected: PASS.

Run: `rtk uv build`

Expected: successful `deeptool-0.4.0` wheel and sdist.

Run: `rtk uv run --extra dashboard deeptool-dashboard --help`

Expected: usage text with `RUNS_DIR`, host, port, refresh, and no-browser options.

- [ ] **Step 4: Commit release preparation**

```bash
rtk git add deeptool/__init__.py tests/test_package.py uv.lock
rtk git commit -m "chore: prepare deeptool 0.4.0"
```

- [ ] **Step 5: Push the reviewed main branch**

Run: `rtk git push origin main`

Expected: remote `main` advances to the local release-preparation commit and CI starts.

Run: `rtk gh pr checks --watch` only if the repository uses a pull request for this
execution; otherwise use `rtk gh run list --workflow CI --limit 1` followed by
`rtk gh run watch RUN_ID` with the returned numeric ID.

Expected: CI and documentation workflows pass before the release is created.

- [ ] **Step 6: Create the GitHub release that triggers trusted PyPI publishing**

Run: `rtk gh release create v0.4.0 --title "deeptool 0.4.0" --generate-notes`

Expected: GitHub publishes release `v0.4.0` and starts the `Publish` workflow.

Run: `rtk gh run list --workflow Publish --limit 1`

Copy the numeric run ID, then run: `rtk gh run watch RUN_ID`

Expected: version/tag check, distribution build, and trusted PyPI publish all pass.

- [ ] **Step 7: Verify the public artifact rather than relying only on workflow status**

After PyPI propagation, create an isolated temporary environment and install
the released dashboard extra:

```bash
rtk uv run --no-project --isolated --with "deeptool[dashboard]==0.4.0" python -c "import deeptool; print(deeptool.__version__)"
```

Expected: `0.4.0`.

Run:

```bash
rtk uvx --from "deeptool[dashboard]==0.4.0" deeptool-dashboard --help
```

Expected: the installed console command prints dashboard usage successfully.
