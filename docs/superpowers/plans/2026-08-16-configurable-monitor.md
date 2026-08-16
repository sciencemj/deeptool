# Configurable Best-Model Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let one exact scalar key and min/max direction jointly control best snapshots, `restore_best()`, and early stopping while preserving the current validation-loss defaults.

**Architecture:** Make `BestSnapshot` metric-neutral, then have `Trainer` assemble a completed epoch row before resolving the configured monitor. `Trainer` owns compatibility-facing properties such as `best_val_loss`, while the snapshot owns only the configured best score and progress coordinate.

**Tech Stack:** Python 3.11+, PyTorch 2.2+, standard-library `math`/typing APIs, pytest 8+

## Global Constraints

- `monitor="val_loss"` and `mode="min"` are the defaults; default epoch behavior must remain unchanged.
- Snapshot selection, `restore_best()`, and early stopping must use the same monitor.
- `ReduceLROnPlateau` must continue receiving `val_loss`, never the custom monitor.
- `best_val_loss` remains the minimum observed validation loss even with a custom monitor.
- Custom metric names remain exact, free-form keys supplied through `Module.log()`.
- Missing custom monitors, NaN, and infinity are errors at the applicable monitor boundary.
- With no validation data, the default monitor and no patience preserve the existing no-best-snapshot behavior.
- Do not add `min_delta`, multi-monitor support, threshold stopping, AMP, or smoothing.

---

## File Structure

- Modify `deeptool/checkpoint.py`: make `BestSnapshot` compare a generic score and persist generic monitor metadata.
- Modify `deeptool/trainer.py`: validate monitor configuration, assemble the row before selection, expose generic best properties, and count non-improving checks.
- Create `tests/test_checkpoint.py`: focused unit coverage for generic snapshot comparison and payload compatibility.
- Modify `tests/test_trainer.py`: integration coverage for custom metrics, early stopping, restoration, and scheduler separation.
- Modify `README.md`, `docs/guide/best.md`, and `docs/guide/best.en.md`: public examples and compatibility semantics.

### Task 1: Make `BestSnapshot` Metric-Neutral

**Files:**
- Create: `tests/test_checkpoint.py`
- Modify: `deeptool/checkpoint.py`

**Interfaces:**
- Consumes: existing `checkpoint_payload()`, `atomic_save()`, model and optimizer state dictionaries.
- Produces: `BestSnapshot(enabled=True, path=None, with_optim=False, monitor="val_loss", mode="min")`; `update(score, progress_name, progress, model, optim) -> bool`; attributes `score`, `progress_name`, and `progress`; `restore(model) -> int`.

- [ ] **Step 1: Write failing generic comparison and compatibility tests**

```python
# tests/test_checkpoint.py
import torch
from torch import nn

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
```

- [ ] **Step 2: Run the focused tests and confirm the old interface fails**

Run: `rtk uv run pytest tests/test_checkpoint.py -q`

Expected: FAIL because `BestSnapshot` does not accept `monitor`/`mode` and `update()` still accepts only validation loss and epoch.

- [ ] **Step 3: Implement the generic snapshot state and comparison**

```python
# deeptool/checkpoint.py
from typing import Literal


class BestSnapshot:
    def __init__(self, enabled: bool = True, path: str | Path | None = None,
                 with_optim: bool = False, monitor: str = "val_loss",
                 mode: Literal["min", "max"] = "min") -> None:
        self.enabled = enabled
        self.path = path
        self.with_optim = with_optim
        self.monitor = monitor
        self.mode = mode
        self.score: float | None = None
        self.progress_name: Literal["epoch", "step"] | None = None
        self.progress: int | None = None
        self._state = None

    def update(self, score: float, progress_name: Literal["epoch", "step"],
               progress: int, model: torch.nn.Module,
               optim: torch.optim.Optimizer) -> bool:
        improved = (
            self.score is None
            or (self.mode == "min" and score < self.score)
            or (self.mode == "max" and score > self.score)
        )
        if not improved:
            return False
        self.score = score
        self.progress_name = progress_name
        self.progress = progress
        # Preserve the existing in-memory and atomic-file branches here.
        return True
```

For a weights-only default `val_loss`/epoch snapshot, keep the exact current
`{"model", "epoch", "val_loss"}` payload. For any other monitor or progress
unit, save `model`, the structural progress key, `monitor`, `mode`, and `score`.
When `with_optim=True`, start with `checkpoint_payload()` and add the generic
fields without removing the existing resumable fields.

Update `restore()` to report the generic monitor, score, and progress unit in
errors and to return `self.progress`. Existing snapshot files remain loadable
because restoration still reads only `payload["model"]`.

- [ ] **Step 4: Run checkpoint tests and existing snapshot tests**

Run: `rtk uv run pytest tests/test_checkpoint.py tests/test_trainer.py -q`

Expected: PASS, including the existing exact default payload assertion.

- [ ] **Step 5: Commit the generic snapshot unit**

```bash
rtk git add deeptool/checkpoint.py tests/test_checkpoint.py tests/test_trainer.py
rtk git commit -m "refactor: generalize best snapshots"
```

### Task 2: Add and Validate the Trainer Monitor API

**Files:**
- Modify: `deeptool/trainer.py`
- Modify: `tests/test_trainer.py`

**Interfaces:**
- Consumes: Task 1's generic `BestSnapshot` and existing `Module.log()` epoch accumulator.
- Produces: constructor args `monitor: str = "val_loss"`, `mode: Literal["min", "max"] = "min"`; properties `best_score`, `best_epoch`, and `best_step`; private `_update_best(row, progress_name, progress) -> bool | None`.

- [ ] **Step 1: Add failing constructor and compatibility-property tests**

```python
# tests/test_trainer.py
def test_monitor_defaults_preserve_validation_loss_behavior():
    trainer = Trainer(max_epochs=4, device="cpu", plot=False)
    trainer.fit(ScriptedLoss([0.5, 0.3, 0.7, 0.9]), ScriptedData())

    assert trainer.monitor == "val_loss"
    assert trainer.mode == "min"
    assert trainer.best_score == pytest.approx(0.3)
    assert trainer.best_epoch == 1
    assert trainer.best_step is None
    assert trainer.best_val_loss == pytest.approx(0.3)


@pytest.mark.parametrize("mode", ["auto", "minimum", "MAX"])
def test_monitor_mode_must_be_min_or_max(mode):
    with pytest.raises(ValueError, match="mode.*min.*max"):
        Trainer(max_epochs=1, mode=mode)


def test_monitor_name_must_not_be_empty():
    with pytest.raises(ValueError, match="monitor"):
        Trainer(max_epochs=1, monitor="")


@pytest.mark.parametrize("monitor", [None, 1, object()])
def test_monitor_name_must_be_a_string(monitor):
    with pytest.raises(ValueError, match="monitor.*string"):
        Trainer(max_epochs=1, monitor=monitor)
```

- [ ] **Step 2: Run the new API tests and confirm they fail**

Run: `rtk uv run pytest tests/test_trainer.py -k 'monitor_defaults or monitor_mode or monitor_name' -q`

Expected: FAIL because the constructor and generic best properties do not exist.

- [ ] **Step 3: Add constructor validation and compatibility properties**

Add the new arguments at the end of the current constructor so existing
positional calls remain stable:

```python
def __init__(self, max_epochs: int,
             device: torch.device | str | None = None,
             gradient_clip_val: float = 0, plot: bool = True,
             snapshot_best: bool = True,
             best_path: str | Path | None = None,
             best_with_optim: bool = False,
             patience: int | None = None,
             log_dir: str | Path | None = None,
             monitor: str = "val_loss",
             mode: Literal["min", "max"] = "min") -> None:
    self.save_hyperparameters()
    if not isinstance(monitor, str) or not monitor:
        raise ValueError("monitor must be a non-empty string")
    if mode not in {"min", "max"}:
        raise ValueError("mode must be 'min' or 'max'")
```

Initialize the snapshot with this exact call:

```python
self._best = BestSnapshot(
    snapshot_best, best_path, best_with_optim,
    monitor=monitor, mode=mode,
)
```

Track
`self._best_val_loss: float | None = None`, and expose:

```python
@property
def best_score(self) -> float | None:
    return self._best.score

@property
def best_epoch(self) -> int | None:
    return self._best.progress if self._best.progress_name == "epoch" else None

@property
def best_step(self) -> int | None:
    return self._best.progress if self._best.progress_name == "step" else None

@property
def best_val_loss(self) -> float | None:
    return self._best_val_loss
```

Update `_best_val_loss` whenever validation loss is finalized. Remove the
snapshot update from `fit_epoch()`; Task 3 will update it only after custom
metrics have been finalized.

- [ ] **Step 4: Run monitor API and existing best-model tests**

Run: `rtk uv run pytest tests/test_trainer.py -k 'best or monitor' -q`

Expected: PASS for constructor and default compatibility tests; custom-monitor tests are not added yet.

- [ ] **Step 5: Commit the public monitor API**

```bash
rtk git add deeptool/trainer.py tests/test_trainer.py
rtk git commit -m "feat: add trainer monitor configuration"
```

### Task 3: Resolve Custom Metrics Before Snapshot and Early Stopping

**Files:**
- Modify: `deeptool/trainer.py`
- Modify: `tests/test_trainer.py`

**Interfaces:**
- Consumes: completed epoch scalar row and generic snapshot from Tasks 1-2.
- Produces: `_epoch_row(metrics, seconds) -> dict[str, object]`,
  `_record_row(row) -> None`, exact-key finite monitor resolution,
  `_bad_monitor_checks` patience state, and persisted rows selected before
  stopping.

- [ ] **Step 1: Add a scripted validation metric and failing behavior tests**

```python
# tests/test_trainer.py
class ScriptedIou(ScriptedLoss):
    def __init__(self, losses, scores):
        super().__init__(losses)
        self.scores = scores

    def validation_step(self, batch):
        index = self.call_count
        loss = super().validation_step(batch)
        self.log("iou", self.scores[index])
        return loss


def test_max_monitor_controls_best_score_epoch_and_early_stopping():
    trainer = Trainer(
        max_epochs=6, device="cpu", plot=False,
        monitor="iou", mode="max", patience=2,
    )
    trainer.fit(
        ScriptedIou([0.5] * 6, [0.4, 0.7, 0.6, 0.5, 0.9, 1.0]),
        ScriptedData(),
    )

    assert trainer.best_score == pytest.approx(0.7)
    assert trainer.best_epoch == 1
    assert trainer.best_val_loss == pytest.approx(0.5)
    assert len(trainer.history["iou"]) == 4


def test_custom_monitor_selects_and_restores_same_snapshot():
    model = ScriptedIou([0.8, 0.7, 0.6], [0.2, 0.9, 0.4])
    trainer = Trainer(
        max_epochs=3, device="cpu", plot=False,
        monitor="iou", mode="max",
    )
    trainer.fit(model, ScriptedData())

    assert trainer.restore_best() == trainer.best_epoch == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_monitor_rejects_non_finite_values(value):
    trainer = Trainer(
        max_epochs=1, device="cpu", plot=False,
        monitor="iou", mode="max",
    )
    with pytest.raises(ValueError, match="iou.*finite.*epoch 0"):
        trainer.fit(ScriptedIou([0.5], [value]), ScriptedData())


def test_missing_custom_monitor_names_epoch():
    trainer = Trainer(max_epochs=1, monitor="iou", mode="max", plot=False)
    with pytest.raises(ValueError, match="iou.*epoch 0"):
        trainer.fit(LinReg(), LinearData())
```

- [ ] **Step 2: Run custom-monitor tests and confirm they fail**

Run: `rtk uv run pytest tests/test_trainer.py -k 'custom_monitor or max_monitor or non_finite or missing_custom' -q`

Expected: FAIL because the best snapshot is no longer updated after removing the validation-loss-only update.

- [ ] **Step 3: Assemble the row, resolve the monitor, and count bad checks**

Refactor the end of the epoch loop to this order:

```python
self.fit_epoch()
metrics = self._finish_logged_scalars()
row = self._epoch_row(metrics, perf_counter() - started)
improved = self._update_best(row, "epoch", self.epoch)
self._record_row(row)
self._step_scheduler()
if self._should_stop_early(improved):
    break
```

Build and record rows with exact helpers so monitor selection and persistence
read the same values:

```python
def _epoch_row(self, metrics: dict[str, float], seconds: float
               ) -> dict[str, object]:
    row: dict[str, object] = {
        "epoch": self.epoch,
        "train_loss": self.history["train_loss"][-1],
    }
    if self.num_val_batches > 0:
        row["val_loss"] = self.history["val_loss"][-1]
    row.update(metrics)
    row["lr"] = float(self.optim.param_groups[0]["lr"])
    row["sec"] = float(seconds)
    return row

def _record_row(self, row: dict[str, object]) -> None:
    if self.recorder is None:
        return
    self.recorder.epoch(**row)
    if not _in_notebook():
        print(" ".join(
            f"{key}={_format_scalar(value)}" for key, value in row.items()
        ))
```

This replaces `_record_epoch()` without changing the `log_dir=None` silence
or notebook output rules.

Use exact lookup and finite validation:

```python
def _update_best(self, row: dict[str, object], progress_name: str,
                 progress: int) -> bool | None:
    if self.monitor == "val_loss" and "val_loss" not in row:
        return None
    if self.monitor not in row:
        raise ValueError(
            f"monitor {self.monitor!r} was not logged at "
            f"{progress_name} {progress}"
        )
    score = float(row[self.monitor])
    if not math.isfinite(score):
        raise ValueError(
            f"monitor {self.monitor!r} must be finite at "
            f"{progress_name} {progress}"
        )
    return self._best.update(
        score, progress_name, progress, self.model, self.optim
    )
```

Replace epoch-distance patience with explicit bad-check counting so the same
mechanism works for step validation later:

```python
def _should_stop_early(self, improved: bool | None) -> bool:
    if self.patience is None or improved is None:
        return False
    self._bad_monitor_checks = 0 if improved else self._bad_monitor_checks + 1
    return self._bad_monitor_checks >= self.patience
```

Keep no-validation/default-monitor training valid when `patience is None`.
Keep the existing early validation-data error when patience is set with the
default monitor. Record the completed row before checking the stop result.

- [ ] **Step 4: Add scheduler-separation and custom snapshot payload tests**

```python
def test_plateau_uses_val_loss_with_custom_monitor(tmp_path):
    model = PlateauScriptedIou(
        losses=[0.5, 0.7, 0.8],
        scores=[0.4, 0.6, 0.7],
    )
    trainer = Trainer(
        max_epochs=3, device="cpu", plot=False,
        log_dir=tmp_path / "exp", monitor="iou", mode="max",
    )
    trainer.fit(model, ScriptedData())

    rows = [
        json.loads(line)
        for line in (tmp_path / "exp/history.jsonl").read_text().splitlines()
    ]
    assert [row["lr"] for row in rows] == pytest.approx([0.1, 0.1, 0.01])


def test_custom_best_file_records_monitor_metadata(tmp_path):
    path = tmp_path / "best.pt"
    trainer = Trainer(
        max_epochs=2, device="cpu", plot=False, best_path=path,
        monitor="iou", mode="max",
    )
    trainer.fit(ScriptedIou([0.5, 0.4], [0.2, 0.8]), ScriptedData())
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["monitor"] == "iou"
    assert payload["mode"] == "max"
    assert payload["score"] == pytest.approx(0.8)
    assert payload["epoch"] == 1
```

Define the plateau helper exactly:

```python
class PlateauScriptedIou(ScriptedIou):
    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", patience=0, factor=0.1
        )
        return optim, scheduler
```

- [ ] **Step 5: Run all Trainer and checkpoint tests**

Run: `rtk uv run pytest tests/test_checkpoint.py tests/test_trainer.py -q`

Expected: PASS, including existing patience timing and exact default checkpoint payload tests.

- [ ] **Step 6: Commit custom monitor selection**

```bash
rtk git add deeptool/checkpoint.py deeptool/trainer.py tests/test_checkpoint.py tests/test_trainer.py
rtk git commit -m "feat: select best models by custom metrics"
```

### Task 4: Document Monitor Semantics and Run Regression Gates

**Files:**
- Modify: `README.md`
- Modify: `docs/guide/best.md`
- Modify: `docs/guide/best.en.md`
- Modify: `docs/guide/trainer.md`
- Modify: `docs/guide/trainer.en.md`

**Interfaces:**
- Consumes: finalized public monitor API from Tasks 1-3.
- Produces: matching Korean/English examples and API descriptions; no new runtime interface.

- [ ] **Step 1: Update Korean and English best-model guides**

Add this exact custom-monitor example in both languages, translated around the
code without changing names:

```python
class SegmentationModel(dt.Module):
    def validation_step(self, batch):
        y_hat = self(*batch[:-1])
        target = batch[-1]
        loss = self.loss(y_hat, target)
        prediction = y_hat.argmax(dim=1)
        intersection = ((prediction == 1) & (target == 1)).sum()
        union = ((prediction == 1) | (target == 1)).sum().clamp_min(1)
        self.log("iou", intersection / union)
        return loss

trainer = dt.Trainer(
    max_epochs=100,
    patience=5,
    monitor="iou",
    mode="max",
    best_path="best.pt",
)
```

Document `best_score`, `best_epoch`, the compatibility meaning of
`best_val_loss`, strict comparisons on ties, missing/non-finite failures, and
the fact that `ReduceLROnPlateau` still sees `val_loss`.

- [ ] **Step 2: Update Trainer reference prose and README API table**

Add `monitor` and `mode` to the constructor table and add `best_score` and
`best_step` to the Trainer API row. Remove the old claim that validation loss
is the only possible best-model criterion; retain it as the default.

- [ ] **Step 3: Run documentation and package-quality tests**

Run: `rtk uv run pytest tests/test_docstrings.py tests/test_package.py tests/test_quickstart.py -q`

Expected: PASS.

Run: `rtk uv run --group docs mkdocs build --strict`

Expected: PASS with no broken links, duplicate anchors, or missing English pages.

- [ ] **Step 4: Run the full regression suite**

Run: `rtk uv run pytest -q`

Expected: PASS on the existing package before beginning the step-training plan.

- [ ] **Step 5: Commit monitor documentation**

```bash
rtk git add README.md docs/guide/best.md docs/guide/best.en.md docs/guide/trainer.md docs/guide/trainer.en.md
rtk git commit -m "docs: explain configurable best monitors"
```
