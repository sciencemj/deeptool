# Run Recording and Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist schema-free epoch metrics as JSONL, compare runs with Matplotlib, support epoch schedulers, and print logged headless progress without changing unlogged behavior.

**Architecture:** A new `deeptool.record` module owns JSON persistence, loading, and comparison figures. `Trainer` owns an epoch-local scalar buffer and optional recorder; `Module.log()` feeds that buffer and reuses the existing board only for live display. Optimizer parsing and scheduler stepping stay inside `Trainer`, immediately after each completed epoch is recorded.

**Tech Stack:** Python 3.11+, standard-library JSON/path/time APIs, PyTorch 2.2+, Matplotlib 3.7+, pytest 8+

## Global Constraints

- `log_dir=None` preserves existing training output and behavior.
- Headless epoch output occurs only when `log_dir` is set and no IPython kernel is active.
- Add no runtime dependency.
- Metric names remain free-form and persist exactly as supplied, except for the
  structural row names `epoch`, `train_loss`, `val_loss`, `lr`, and `sec`.
- JSONL is appended one completed epoch at a time and flushed before returning.
- Support epoch-stepped PyTorch schedulers and `ReduceLROnPlateau`; do not add AMP or batch-stepped scheduler support.

---

## File structure

- Create `deeptool/record.py`: scalar normalization, `RunRecorder`, `load_runs`, and `plot_runs`.
- Modify `deeptool/__init__.py`: export the three recording APIs.
- Modify `deeptool/module.py`: add `Module.log()` and widen the optimizer configuration return annotation.
- Modify `deeptool/trainer.py`: attach recorder, aggregate epoch metrics, record metadata/history, print headless rows, and step schedulers.
- Create `tests/test_record.py`: persistence, loading, plotting, and malformed-input contracts.
- Modify `tests/test_module.py`: custom metric logging and validation.
- Modify `tests/test_trainer.py`: end-to-end persistence, failure survival, headless behavior, board behavior, and scheduler contracts.
- Modify `README.md`, `docs/api.md`, `docs/guide/trainer.md`, and `docs/guide/trainer.en.md`: public usage and limitations.

### Task 1: Standalone run records and comparison API

**Files:**
- Create: `deeptool/record.py`
- Create: `tests/test_record.py`
- Modify: `deeptool/__init__.py`

**Interfaces:**
- Produces: `RunRecorder(log_dir: str | Path)`, `RunRecorder.meta(**info)`, `RunRecorder.epoch(**scalars)`, `load_runs(root: str | Path) -> dict[str, dict[str, list]]`, and `plot_runs(source) -> list[matplotlib.figure.Figure]`.
- Consumes: only standard-library JSON/filesystem functions and Matplotlib.

- [ ] **Step 1: Write failing recorder tests**

```python
# tests/test_record.py
import json

from matplotlib.figure import Figure
import pytest

from deeptool.record import RunRecorder, load_runs, plot_runs


def test_recorder_writes_metadata_and_append_only_epoch_rows(tmp_path):
    recorder = RunRecorder(tmp_path / "exp1")
    recorder.meta(device="cpu", model_class="tests.Model")
    recorder.epoch(epoch=0, train_loss=2.0, iou=0.3)

    first_read = (tmp_path / "exp1" / "history.jsonl").read_text()
    assert json.loads(first_read) == {
        "epoch": 0, "train_loss": 2.0, "iou": 0.3
    }

    recorder.epoch(epoch=1, train_loss=1.0, iou=0.5)
    rows = [json.loads(line) for line in
            (tmp_path / "exp1" / "history.jsonl").read_text().splitlines()]
    assert rows == [
        {"epoch": 0, "train_loss": 2.0, "iou": 0.3},
        {"epoch": 1, "train_loss": 1.0, "iou": 0.5},
    ]
    assert json.loads((tmp_path / "exp1" / "meta.json").read_text()) == {
        "device": "cpu", "model_class": "tests.Model"
    }


def test_load_runs_aligns_sparse_metrics_and_heterogeneous_runs(tmp_path):
    a = RunRecorder(tmp_path / "a")
    a.epoch(epoch=0, iou=0.2)
    a.epoch(epoch=1, acc50=0.7)
    b = RunRecorder(tmp_path / "b")
    b.epoch(epoch=0, ap50=0.4, recall=0.6)

    assert load_runs(tmp_path) == {
        "a": {"epoch": [0, 1], "iou": [0.2, None],
              "acc50": [None, 0.7]},
        "b": {"epoch": [0], "ap50": [0.4], "recall": [0.6]},
    }


def test_load_runs_reports_malformed_line_location(tmp_path):
    run = tmp_path / "broken"
    run.mkdir()
    (run / "history.jsonl").write_text('{"epoch": 0}\nnot-json\n')

    with pytest.raises(ValueError, match=r"broken/history.jsonl.*line 2"):
        load_runs(tmp_path)


def test_plot_runs_returns_one_figure_per_metric(tmp_path):
    RunRecorder(tmp_path / "a").epoch(epoch=0, iou=0.2, acc50=0.7)
    RunRecorder(tmp_path / "b").epoch(epoch=0, ap50=0.4)

    figures = plot_runs(tmp_path)

    assert len(figures) == 3
    assert all(isinstance(figure, Figure) for figure in figures)
    assert {figure.axes[0].get_ylabel() for figure in figures} == {
        "iou", "acc50", "ap50"
    }
```

- [ ] **Step 2: Run the focused tests and verify import failure**

Run: `uv run pytest tests/test_record.py -q`

Expected: collection fails because `deeptool.record` does not exist.

- [ ] **Step 3: Implement the minimal record module and exports**

Implement `RunRecorder` with `mkdir(parents=True, exist_ok=True)`, an atomic
`meta.json.tmp` replacement, and one `open("a", encoding="utf-8")` call per
epoch followed by `flush()`. Implement `load_runs` by reading sorted immediate
child directories and building every metric list with
`[row.get(key) for row in rows]`. Wrap JSON decode failures as:

```python
raise ValueError(f"invalid JSON in {path} at line {line_number}") from error
```

Implement `plot_runs` by accepting either a path or loaded mapping, preserving
first-seen metric order, excluding `epoch`, and plotting `run["epoch"]` against
the aligned values. Add `RunRecorder`, `load_runs`, and `plot_runs` to
`deeptool.__all__`.

- [ ] **Step 4: Run focused tests and public-symbol tests**

Run: `uv run pytest tests/test_record.py tests/test_package.py tests/test_docstrings.py -q`

Expected: all pass.

- [ ] **Step 5: Commit the standalone record API**

```bash
git add deeptool/record.py deeptool/__init__.py tests/test_record.py
git commit -m "feat: add persistent run records"
```

### Task 2: Model scalar logging and epoch aggregation

**Files:**
- Modify: `deeptool/module.py:35-70`
- Modify: `deeptool/trainer.py:71-75,113-153`
- Modify: `tests/test_module.py`
- Modify: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `Module.trainer`, `Module.plot(key, value, train)`, and the existing Trainer epoch loop.
- Produces: `Module.log(key: str, value: torch.Tensor | float) -> None` and private `Trainer._log_scalar(key: str, value: float) -> None`.

- [ ] **Step 1: Write failing custom-metric tests**

Add a `MetricLinReg` test model whose `training_step` calls
`self.log("iou", torch.tensor(0.2))` and `self.log("iou", 0.4)` before returning
the real loss. Add tests:

```python
def test_log_averages_arbitrary_scalars_per_epoch():
    trainer = Trainer(max_epochs=2, device="cpu", plot=False)
    trainer.fit(MetricLinReg(), LinearData())

    assert trainer.history["iou"] == pytest.approx([0.3, 0.3])


def test_log_uses_original_name_for_history_and_phase_prefix_for_board():
    trainer = Trainer(max_epochs=1, device="cpu", plot=True)
    trainer.fit(MetricLinReg(), LinearData())

    assert "iou" in trainer.history
    assert "train_iou" in trainer.board.data


def test_log_rejects_non_scalar_tensors():
    model = ToyNet()
    model.trainer = FakeTrainer()

    with pytest.raises(ValueError, match="scalar"):
        model.log("iou", torch.tensor([0.2, 0.4]))
```

- [ ] **Step 2: Run focused tests and verify missing method failure**

Run: `uv run pytest tests/test_module.py tests/test_trainer.py -q`

Expected: the new tests fail because `Module.log` does not exist.

- [ ] **Step 3: Implement minimal scalar conversion and aggregation**

In `Module.log`, return immediately without an attached Trainer, convert a
scalar tensor with `detach().cpu().item()`, reject tensors with `numel() != 1`,
convert numeric values to `float`, call `trainer._log_scalar(key, value)`, and
then call `plot(key, value, train=self.training)`.

In `Trainer`, reset `_epoch_scalars: dict[str, list[float]]` before every epoch.
At epoch completion, average every list. Append `None` to an existing custom
history key when it is absent in the current epoch; initialize a newly seen key
with `[None] * self.epoch` before its average. Keep the existing built-in loss
lists unchanged. Reserve `epoch`, `train_loss`, `val_loss`, `lr`, and `sec` for
Trainer rows and raise `ValueError` if `Module.log` uses one of them.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/test_module.py tests/test_trainer.py -q`

Expected: all pass.

- [ ] **Step 5: Commit model logging**

```bash
git add deeptool/module.py deeptool/trainer.py tests/test_module.py tests/test_trainer.py
git commit -m "feat: aggregate custom epoch metrics"
```

### Task 3: Trainer persistence and headless progress

**Files:**
- Modify: `deeptool/trainer.py:1-153`
- Modify: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `RunRecorder`, epoch averages from Task 2, `_in_notebook()`, Trainer/model hyperparameters, and optimizer parameter groups.
- Produces: `Trainer(..., log_dir: str | Path | None = None)`, `meta.json`, `history.jsonl`, and conditional progress lines.

- [ ] **Step 1: Write failing end-to-end recording tests**

```python
def test_log_dir_records_metadata_and_complete_epoch_rows(tmp_path, capsys):
    log_dir = tmp_path / "exp1"
    trainer = Trainer(max_epochs=2, device="cpu", plot=False, log_dir=log_dir)
    trainer.fit(MetricLinReg(), LinearData())

    meta = json.loads((log_dir / "meta.json").read_text())
    rows = [json.loads(line) for line in
            (log_dir / "history.jsonl").read_text().splitlines()]

    assert meta["device"] == "cpu"
    assert meta["model_class"].endswith(".MetricLinReg")
    assert len(rows) == 2
    assert list(rows[0]) == ["epoch", "train_loss", "val_loss", "iou", "lr", "sec"]
    assert rows[0]["epoch"] == 0
    assert rows[0]["iou"] == pytest.approx(0.3)
    assert rows[0]["lr"] == pytest.approx(0.1)
    assert rows[0]["sec"] >= 0
    assert "epoch=0" in capsys.readouterr().out


def test_log_dir_none_remains_silent(capsys):
    Trainer(max_epochs=1, device="cpu", plot=False).fit(LinReg(), LinearData())
    assert capsys.readouterr().out == ""


def test_logged_notebook_run_does_not_print(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("deeptool.trainer._in_notebook", lambda: True)
    trainer = Trainer(max_epochs=1, device="cpu", plot=False,
                      log_dir=tmp_path / "exp")
    trainer.fit(LinReg(), LinearData())

    assert capsys.readouterr().out == ""


def test_completed_rows_survive_a_later_training_failure(tmp_path):
    trainer = Trainer(max_epochs=3, device="cpu", plot=False,
                      log_dir=tmp_path / "exp")

    with pytest.raises(RuntimeError, match="planned failure"):
        trainer.fit(FailsOnSecondEpoch(), LinearData())

    rows = (tmp_path / "exp" / "history.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["epoch"] == 0


def test_script_board_accumulates_without_creating_figures(tmp_path):
    trainer = Trainer(max_epochs=1, device="cpu", plot=True,
                      log_dir=tmp_path / "exp")
    trainer.fit(LinReg(), LinearData())

    assert trainer.board.data["train_loss"]
    assert trainer.board.fig is None
```

The failure model raises in `training_step` when `self.trainer.epoch == 1` and
otherwise delegates to `super().training_step(batch)`.

- [ ] **Step 2: Run focused tests and verify constructor/API failure**

Run: `uv run pytest tests/test_trainer.py -q`

Expected: new tests fail because `Trainer` has no `log_dir` argument.

- [ ] **Step 3: Implement recording lifecycle and output**

Create `self.recorder` only when `log_dir` is non-`None`. Construct a board with
`display=_in_notebook()` so headless scripts retain data without rendering.
After the model and optimizer exist, call `recorder.meta()` with a UTC ISO
timestamp, resolved device, fully qualified model class, and trainer/model
hparams; serialize unsupported metadata values through `json.dumps` with
`default=str` in the recorder.

Measure each epoch with `time.perf_counter()`. Build rows in the exact order
`epoch`, `train_loss`, optional `val_loss`, custom metrics, `lr`, `sec`; write
the row before early stopping. When a recorder exists and `_in_notebook()` is
false, print one compact `key=value` line from the same row. Do not print from
any path when no recorder exists.

- [ ] **Step 4: Run focused and regression tests**

Run: `uv run pytest tests/test_trainer.py tests/test_quickstart.py -q`

Expected: all pass, including old loss history and early stopping tests.

- [ ] **Step 5: Commit Trainer persistence**

```bash
git add deeptool/trainer.py tests/test_trainer.py
git commit -m "feat: persist trainer epoch history"
```

### Task 4: Epoch scheduler support

**Files:**
- Modify: `deeptool/module.py:35-36`
- Modify: `deeptool/trainer.py:113-153`
- Modify: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `Module.configure_optimizers()` and completed epoch `val_loss`.
- Produces: optimizer-only or `(optimizer, scheduler)` configuration, `trainer.scheduler`, and post-record scheduler stepping.

- [ ] **Step 1: Write failing scheduler tests**

```python
class StepScheduledLinReg(LinReg):
    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.1)
        return optim, torch.optim.lr_scheduler.StepLR(optim, step_size=1,
                                                       gamma=0.1)


def test_epoch_scheduler_changes_next_epochs_recorded_lr(tmp_path):
    trainer = Trainer(max_epochs=2, device="cpu", plot=False,
                      log_dir=tmp_path / "exp")
    trainer.fit(StepScheduledLinReg(), LinearData())

    rows = [json.loads(line) for line in
            (tmp_path / "exp" / "history.jsonl").read_text().splitlines()]
    assert [row["lr"] for row in rows] == pytest.approx([0.1, 0.01])


class PlateauLinReg(LinReg):
    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", patience=0, factor=0.1
        )
        return optim, scheduler


def test_plateau_scheduler_uses_validation_loss(tmp_path):
    trainer = Trainer(max_epochs=3, device="cpu", plot=False,
                      log_dir=tmp_path / "exp")
    trainer.fit(PlateauScriptedLoss([0.5, 0.7, 0.8]), ScriptedData())

    rows = [json.loads(line) for line in
            (tmp_path / "exp" / "history.jsonl").read_text().splitlines()]
    assert [row["lr"] for row in rows] == pytest.approx([0.1, 0.1, 0.01])


def test_plateau_scheduler_requires_validation_data():
    trainer = Trainer(max_epochs=1, device="cpu", plot=False)
    with pytest.raises(ValueError, match="validation data"):
        trainer.fit(PlateauLinReg(), NoValData())
```

`PlateauScriptedLoss` combines the existing scripted validation loss behavior
with `PlateauLinReg.configure_optimizers`.

- [ ] **Step 2: Run scheduler tests and verify tuple failure**

Run: `uv run pytest tests/test_trainer.py -q`

Expected: scheduler tests fail because Trainer treats the tuple as an optimizer.

- [ ] **Step 3: Implement optimizer parsing and scheduler dispatch**

Accept either a `torch.optim.Optimizer` or a two-item tuple. Store the optional
scheduler on `self.scheduler`. Reject any other tuple length with `ValueError`.
Before the epoch loop, reject `ReduceLROnPlateau` when no validation loader
exists. After recording/printing each epoch, call:

```python
if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    self.scheduler.step(self.history["val_loss"][-1])
elif self.scheduler is not None:
    self.scheduler.step()
```

Update `Module.configure_optimizers` documentation and annotation to describe
the supported union and the exclusion of batch-stepped schedulers.

- [ ] **Step 4: Run all Trainer and Module tests**

Run: `uv run pytest tests/test_trainer.py tests/test_module.py -q`

Expected: all pass.

- [ ] **Step 5: Commit scheduler support**

```bash
git add deeptool/module.py deeptool/trainer.py tests/test_module.py tests/test_trainer.py
git commit -m "feat: support epoch lr schedulers"
```

### Task 5: User documentation and full verification

**Files:**
- Modify: `README.md`
- Modify: `docs/api.md`
- Modify: `docs/guide/trainer.md`
- Modify: `docs/guide/trainer.en.md`

**Interfaces:**
- Documents: `log_dir`, JSONL schema, `Module.log`, `load_runs`, `plot_runs`, headless output condition, optimizer/scheduler tuple, and unsupported AMP/batch scheduling.

- [ ] **Step 1: Update Korean and English documentation**

Add a concise persistent-runs section with this executable API shape:

```python
trainer = dt.Trainer(max_epochs=50, log_dir="runs/exp1", plot=False)
trainer.fit(model, data)

runs = dt.load_runs("runs")
figures = dt.plot_runs(runs)
```

Document `self.log("iou", value)`, exact persisted names, epoch averaging,
metadata/history filenames, conditional headless output, scheduler tuple
configuration, `ReduceLROnPlateau` validation requirement, and the exclusion of
AMP/`OneCycleLR`. Add `deeptool.record` members to `docs/api.md` and public
symbols to README's API table.

- [ ] **Step 2: Run fresh full verification**

Run: `uv run pytest -q`

Expected: zero failures.

Run: `uv run mkdocs build --strict`

Expected: exit code 0 with no documentation warnings.

Run: `git diff --check`

Expected: no output and exit code 0.

- [ ] **Step 3: Review the implementation against all acceptance criteria**

Confirm from test names and fresh output that unlogged behavior is silent,
completed lines survive later failure, heterogeneous metrics load and plot,
`plot=False + log_dir` records, normal and plateau schedulers use correct timing,
and AMP remains untouched.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md docs/api.md docs/guide/trainer.md docs/guide/trainer.en.md
git commit -m "docs: explain recorded training runs"
```
