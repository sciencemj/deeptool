# Step-Based Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a finite-DataLoader training mode bounded by optimizer steps, with interval records, validation, schedulers, best-model monitoring, and step-aware checkpoints and plots.

**Architecture:** Keep the epoch loop as the compatibility path while extracting one shared optimizer-update primitive. A separate step loop repeats the train loader, emits rows at log/validation/final boundaries, and invokes the monitor only at configured check boundaries.

**Tech Stack:** Python 3.11+, PyTorch 2.2+ DataLoader and LR schedulers, JSONL records, Matplotlib 3.7+, pytest 8+

## Global Constraints

- Complete the configurable-monitor plan first; this plan consumes
  `BestSnapshot.update(score, progress_name, progress, model, optim) -> bool`,
  `best_score`, and generic patience counting.
- Exactly one positive integer of `max_epochs` and `max_steps` must be supplied; bool values are not valid integers.
- A step is one completed `optimizer.step()` and persisted step values are one-based.
- `log_every_n_steps` defaults to `1`; `val_every_n_steps` defaults to `None`, meaning final-step validation only.
- A finite non-empty train DataLoader is required; lengthless/infinite IterableDatasets are out of scope.
- Step-mode patience counts monitor checks, not optimizer updates.
- `scheduler_interval="auto"` resolves to the active training unit.
- `ReduceLROnPlateau` remains validation-loss-driven and ignores the normal interval.
- Do not add AMP, gradient accumulation, distributed training, token-count scheduling, or automatic resume.
- Existing epoch histories, output, scheduler timing, checkpoints, and public calls must remain compatible.

---

## File Structure

- Modify `deeptool/trainer.py`: mode validation, shared optimizer update, step loop, interval rows, validation checks, scheduler timing, progress properties, and step checkpoints.
- Modify `deeptool/module.py`: choose notebook plot x coordinates from the Trainer's active unit.
- Modify `deeptool/checkpoint.py`: optionally include and return a global step in full checkpoints.
- Modify `deeptool/record.py`: treat `step` as a structural x key when plotting.
- Create `tests/test_step_training.py`: isolated step-mode loop, cadence, monitor, scheduler, record, and failure tests.
- Modify `tests/test_trainer.py`, `tests/test_module.py`, and `tests/test_record.py`: epoch regression and shared API checks.
- Modify Korean/English Trainer and best-model guides plus README.

### Task 1: Add Exclusive Training-Unit Configuration

**Files:**
- Create: `tests/test_step_training.py`
- Modify: `deeptool/trainer.py`

**Interfaces:**
- Consumes: current Trainer constructor and `HyperParameters.save_hyperparameters()`.
- Produces: optional `max_epochs`; args `max_steps`, `log_every_n_steps`, `val_every_n_steps`, `scheduler_interval`; properties `training_unit` and `global_step`.

- [ ] **Step 1: Write failing constructor validation tests**

```python
# tests/test_step_training.py
import pytest

from deeptool.trainer import Trainer


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "exactly one"),
        ({"max_epochs": 1, "max_steps": 2}, "exactly one"),
        ({"max_steps": 0}, "max_steps"),
        ({"max_steps": True}, "max_steps"),
        ({"max_epochs": False}, "max_epochs"),
        ({"max_steps": 2, "log_every_n_steps": 0}, "log_every"),
        ({"max_steps": 2, "val_every_n_steps": 0}, "val_every"),
        ({"max_steps": 2, "scheduler_interval": "batch"}, "scheduler_interval"),
    ],
)
def test_training_limit_and_interval_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Trainer(**kwargs)


def test_training_unit_is_derived_from_the_selected_limit():
    assert Trainer(max_epochs=2).training_unit == "epoch"
    trainer = Trainer(max_steps=3)
    assert trainer.training_unit == "step"
    assert trainer.global_step == 0
```

- [ ] **Step 2: Run constructor tests and confirm they fail**

Run: `rtk uv run pytest tests/test_step_training.py -q`

Expected: FAIL because `max_epochs` is required and the new arguments do not exist.

- [ ] **Step 3: Add the backward-compatible constructor signature and validation helpers**

Keep `max_epochs` first and give it a default; append all new parameters after
the previously added monitor parameters:

```python
def __init__(
    self,
    max_epochs: int | None = None,
    device: torch.device | str | None = None,
    gradient_clip_val: float = 0,
    plot: bool = True,
    snapshot_best: bool = True,
    best_path: str | Path | None = None,
    best_with_optim: bool = False,
    patience: int | None = None,
    log_dir: str | Path | None = None,
    monitor: str = "val_loss",
    mode: Literal["min", "max"] = "min",
    max_steps: int | None = None,
    log_every_n_steps: int = 1,
    val_every_n_steps: int | None = None,
    scheduler_interval: Literal["auto", "epoch", "step"] = "auto",
) -> None:
```

Use a helper that rejects bool explicitly:

```python
def _positive_int(name: str, value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return True
```

Validate exactly one non-`None` training limit, positive logging intervals,
optional positive validation interval, and the three scheduler interval
strings. Set `training_unit`, `global_step = 0`, and keep `epoch = 0` for epoch
compatibility and completed train-loader pass tracking.

- [ ] **Step 4: Run constructor and existing hyperparameter tests**

Run: `rtk uv run pytest tests/test_step_training.py tests/test_core.py tests/test_trainer.py -k 'training_limit or training_unit or hparam or explicit_device' -q`

Expected: PASS; existing calls such as `Trainer(max_epochs=2)` still construct normally.

- [ ] **Step 5: Commit training-unit configuration**

```bash
rtk git add deeptool/trainer.py tests/test_step_training.py
rtk git commit -m "feat: configure step-based training"
```

### Task 2: Extract One Shared Optimizer Update

**Files:**
- Modify: `deeptool/trainer.py`
- Modify: `tests/test_trainer.py`
- Modify: `tests/test_step_training.py`

**Interfaces:**
- Consumes: prepared batches, model `training_step()`, optimizer, gradient clipping.
- Produces: `_train_batch(batch) -> tuple[float, float]`, returning detached loss and the learning rate used by that update; `_resolved_scheduler_interval() -> Literal["epoch", "step"]`.

- [ ] **Step 1: Add a regression test for update count and LR capture**

```python
# tests/test_step_training.py
import torch
from torch import nn
from torch.nn import functional as F

from deeptool.data import DataModule
from deeptool.module import Module


class TinyData(DataModule):
    def __init__(self, train_batches=2, val_batches=1):
        super().__init__(batch_size=1)
        self.X = torch.arange(train_batches + val_batches, dtype=torch.float32).view(-1, 1)
        self.y = 2 * self.X
        self.train_batches = train_batches
        self.val_batches = val_batches

    def get_dataloader(self, train):
        if train:
            return self.get_tensorloader(
                (self.X, self.y), True, slice(0, self.train_batches)
            )
        if self.val_batches == 0:
            return None
        return self.get_tensorloader(
            (self.X, self.y), False,
            slice(self.train_batches, self.train_batches + self.val_batches),
        )


class CountingModel(Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1, 1)
        self.updates = 0

    def loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)

    def training_step(self, batch):
        self.updates += 1
        return super().training_step(batch)


def test_step_mode_performs_exactly_max_steps_updates():
    model = CountingModel()
    trainer = Trainer(max_steps=5, device="cpu", plot=False)
    trainer.fit(model, TinyData(train_batches=2, val_batches=0))

    assert model.updates == 5
    assert trainer.global_step == 5
```

- [ ] **Step 2: Run the exact-step test and confirm it fails**

Run: `rtk uv run pytest tests/test_step_training.py::test_step_mode_performs_exactly_max_steps_updates -q`

Expected: FAIL because `fit()` still ranges over `max_epochs`.

- [ ] **Step 3: Extract `_train_batch()` without changing epoch behavior**

```python
def _train_batch(self, batch: Sequence[torch.Tensor]) -> tuple[float, float]:
    loss = self.model.training_step(self.prepare_batch(batch))
    self.optim.zero_grad()
    loss.backward()
    if self.gradient_clip_val > 0:
        self.clip_gradients(self.gradient_clip_val)
    lr_used = float(self.optim.param_groups[0]["lr"])
    self.optim.step()
    self.train_batch_idx += 1
    self.global_step += 1
    if self._resolved_scheduler_interval() == "step":
        self._step_normal_scheduler()
    return float(loss.detach().cpu().item()), lr_used
```

`_step_normal_scheduler()` must skip `None` and `ReduceLROnPlateau`. Rewrite
the current epoch training loop to call `_train_batch()` and keep the returned
losses. In epoch mode, `global_step` may now be inspected but does not change
history or recording schemas.

Implement `_fit_steps()` as the first bounded-optimization slice: repeat the
finite loader and call `_train_batch()` until `global_step == max_steps`.
Interval row emission and validation are a separate, independently tested
responsibility in Task 3.

- [ ] **Step 4: Run all epoch Trainer tests plus the exact-step test**

Run: `rtk uv run pytest tests/test_trainer.py tests/test_step_training.py::test_step_mode_performs_exactly_max_steps_updates -q`

Expected: PASS; especially gradient clipping, batch counters, lazy parameters, and epoch loss regression tests.

- [ ] **Step 5: Commit the shared optimizer primitive**

```bash
rtk git add deeptool/trainer.py tests/test_trainer.py tests/test_step_training.py
rtk git commit -m "refactor: share optimizer update logic"
```

### Task 3: Emit Step Rows at Logging, Validation, and Final Boundaries

**Files:**
- Modify: `deeptool/trainer.py`
- Modify: `tests/test_step_training.py`

**Interfaces:**
- Consumes: Task 2's `_train_batch()` and existing `RunRecorder`.
- Produces: `_fit_steps()`, `_run_validation() -> float | None`,
  `_step_row(train_losses, val_loss, metrics, lr, seconds) -> dict[str, object]`,
  step-aligned `Trainer.history`, and rows keyed by one-based `step`.

- [ ] **Step 1: Write failing DataLoader repetition and row-cadence tests**

```python
# tests/test_step_training.py
import json


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_step_mode_repeats_loader_and_records_log_boundaries(tmp_path):
    trainer = Trainer(
        max_steps=5, log_every_n_steps=2,
        device="cpu", plot=False, log_dir=tmp_path / "exp",
    )
    trainer.fit(CountingModel(), TinyData(train_batches=2, val_batches=0))

    rows = _rows(tmp_path / "exp/history.jsonl")
    assert [row["step"] for row in rows] == [2, 4, 5]
    assert [len(trainer.history[key]) for key in ("step", "train_loss")] == [3, 3]
    assert trainer.history["step"] == [2, 4, 5]
    assert all("epoch" not in row for row in rows)


def test_validation_boundary_forces_one_row_without_duplicates(tmp_path):
    trainer = Trainer(
        max_steps=6, log_every_n_steps=4, val_every_n_steps=3,
        device="cpu", plot=False, log_dir=tmp_path / "exp",
    )
    trainer.fit(CountingModel(), TinyData())

    rows = _rows(tmp_path / "exp/history.jsonl")
    assert [row["step"] for row in rows] == [3, 4, 6]
    assert ["val_loss" in row for row in rows] == [True, False, True]


def test_no_validation_interval_runs_validation_only_at_final_step(tmp_path):
    trainer = Trainer(
        max_steps=3, device="cpu", plot=False,
        log_dir=tmp_path / "exp",
    )
    trainer.fit(CountingModel(), TinyData())

    rows = _rows(tmp_path / "exp/history.jsonl")
    assert [row["step"] for row in rows] == [1, 2, 3]
    assert ["val_loss" in row for row in rows] == [False, False, True]
```

- [ ] **Step 2: Run cadence tests and confirm they fail**

Run: `rtk uv run pytest tests/test_step_training.py -k 'records_log_boundaries or validation_boundary or final_step' -q`

Expected: FAIL because the bounded update slice does not yet emit step records or run validation.

- [ ] **Step 3: Implement interval accumulators and validation extraction**

Track interval state locally inside `_fit_steps()`:

```python
losses: list[float] = []
window_started = perf_counter()
last_lr = float(self.optim.param_groups[0]["lr"])

while self.global_step < self.max_steps:
    try:
        batch = next(train_iterator)
    except StopIteration:
        self.epoch += 1
        if self._resolved_scheduler_interval() == "epoch":
            self._step_normal_scheduler()
        train_iterator = iter(self.train_dataloader)
        batch = next(train_iterator)

    loss, last_lr = self._train_batch(batch)
    losses.append(loss)
    validate = (
        self.global_step == self.max_steps
        or self.val_every_n_steps is not None
        and self.global_step % self.val_every_n_steps == 0
    )
    emit = (
        validate
        or self.global_step % self.log_every_n_steps == 0
        or self.global_step == self.max_steps
    )
```

Extract validation from `fit_epoch()` into `_run_validation()` so both modes
share model eval/no-grad, loss averaging, `val_batch_idx`, and custom logging.
Epoch mode calls it once per epoch and keeps its existing history alignment.

When `emit` is true, finalize custom scalars once, assemble a row starting with
`step`, average the `losses` window, include `val_loss` only when validation ran,
include `last_lr` and window seconds, align every step-history key with `None`,
record/print the row, then reset losses/scalars/window time. A coincident log
and validation boundary passes through this branch once.

Use one pure row assembler before history and recorder side effects:

```python
def _step_row(
    self,
    train_losses: list[float],
    val_loss: float | None,
    metrics: dict[str, float],
    lr: float,
    seconds: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "step": self.global_step,
        "train_loss": sum(train_losses) / len(train_losses),
    }
    if val_loss is not None:
        row["val_loss"] = val_loss
    row.update(metrics)
    row["lr"] = lr
    row["sec"] = seconds
    return row
```

Task 3 appends this row to step history and then calls the monitor plan's
`_record_row(row)`. Task 4 inserts monitor and scheduler decisions between row
assembly and persistence without changing the schema.

Keep epoch history behavior by adding an `append_history` switch to scalar
finalization. Epoch mode uses the existing `True` behavior; step mode requests
averages without mutating history, then aligns the complete step row itself:

```python
def _finish_logged_scalars(
    self, *, append_history: bool = True
) -> dict[str, float]:
    metrics = {
        key: sum(values) / len(values)
        for key, values in self._epoch_scalars.items()
    }
    if append_history:
        self._append_epoch_metrics(metrics)
    return metrics

def _append_epoch_metrics(self, metrics: dict[str, float]) -> None:
    for key in self._logged_metric_names:
        if key not in metrics:
            self.history[key].append(None)
    for key, value in metrics.items():
        if key not in self._logged_metric_names:
            self._logged_metric_names.append(key)
            self.history[key] = [None] * self.epoch
        self.history[key].append(value)

def _append_step_history(self, row: dict[str, object]) -> None:
    history_keys = {
        key for key in row
        if key not in {"lr", "sec"}
    }
    previous = len(self.history["step"])
    for key in history_keys - self.history.keys():
        self.history[key] = [None] * previous
    for key in self.history:
        self.history[key].append(row.get(key))
```

Initialize step history as
`{"step": [], "train_loss": [], "val_loss": []}` and epoch history exactly as
today. Call `_finish_logged_scalars(append_history=False)` only when a step row
is emitted, then `_append_step_history(row)` once. This preserves custom metric
x coordinates at forced validation rows.

- [ ] **Step 4: Reject empty and lengthless train loaders explicitly**

Add these exact test helpers and assertions:

```python
from torch.utils import data as torch_data


class EmptyData(DataModule):
    def get_dataloader(self, train):
        if not train:
            return None
        dataset = torch_data.TensorDataset(
            torch.empty(0, 1), torch.empty(0, 1)
        )
        return torch_data.DataLoader(dataset, batch_size=1, shuffle=False)


class LengthlessLoader:
    def __iter__(self):
        batch = (torch.zeros(1, 1), torch.zeros(1, 1))
        return iter([batch])


class LengthlessData(DataModule):
    def get_dataloader(self, train):
        return LengthlessLoader() if train else None


def test_step_mode_rejects_empty_train_loader():
    with pytest.raises(ValueError, match="train dataloader.*empty"):
        Trainer(max_steps=1, plot=False).fit(CountingModel(), EmptyData())


def test_step_mode_rejects_lengthless_train_loader():
    with pytest.raises(ValueError, match="finite.*length"):
        Trainer(max_steps=1, plot=False).fit(CountingModel(), LengthlessData())
```

In `prepare_data()`, catch `TypeError` from `len(train_dataloader)` only for
step mode and raise the documented finite-loader error. Reject length zero
before lazy-parameter materialization.

- [ ] **Step 5: Run cadence, failure, record, and epoch regression tests**

Run: `rtk uv run pytest tests/test_step_training.py tests/test_trainer.py tests/test_record.py -q`

Expected: PASS; epoch JSONL still starts with zero-based `epoch` and step JSONL with one-based `step`.

- [ ] **Step 6: Commit interval rows and validation**

```bash
rtk git add deeptool/trainer.py tests/test_step_training.py tests/test_trainer.py
rtk git commit -m "feat: record interval-based training steps"
```

### Task 4: Apply Monitor and Patience at Step Check Boundaries

**Files:**
- Modify: `deeptool/trainer.py`
- Modify: `tests/test_step_training.py`

**Interfaces:**
- Consumes: configurable-monitor plan's `_update_best(row, progress_name, progress)` and `_should_stop_early(improved)`.
- Produces: step-mode `best_step`, custom monitor snapshots, and patience measured in check boundaries.

- [ ] **Step 1: Add a scripted step-validation model and failing monitor tests**

```python
# tests/test_step_training.py
class StepMetricModel(CountingModel):
    def __init__(self, scores):
        super().__init__()
        self.scores = iter(scores)

    def validation_step(self, batch):
        loss = super().validation_step(batch)
        self.log("iou", next(self.scores))
        return loss


def test_step_monitor_tracks_best_step_and_restore_value():
    model = StepMetricModel([0.4, 0.8, 0.6])
    trainer = Trainer(
        max_steps=6, val_every_n_steps=2,
        monitor="iou", mode="max", device="cpu", plot=False,
    )
    trainer.fit(model, TinyData())

    assert trainer.best_score == pytest.approx(0.8)
    assert trainer.best_step == 4
    assert trainer.best_epoch is None
    assert trainer.restore_best() == 4


def test_step_patience_counts_validation_checks_not_steps():
    model = StepMetricModel([0.8, 0.7, 0.6, 0.9])
    trainer = Trainer(
        max_steps=10, val_every_n_steps=2, patience=2,
        monitor="iou", mode="max", device="cpu", plot=False,
    )
    trainer.fit(model, TinyData())

    assert trainer.global_step == 6
    assert trainer.history["step"][-1] == 6
    assert trainer.best_step == 2
```

- [ ] **Step 2: Run step-monitor tests and confirm they fail**

Run: `rtk uv run pytest tests/test_step_training.py -k 'step_monitor or step_patience' -q`

Expected: FAIL because `_fit_steps()` records rows but does not call the monitor at validation boundaries.

- [ ] **Step 3: Invoke the generic monitor only at check boundaries**

After assembling a validation/check row and before scheduler/stop handling:

```python
improved = self._update_best(row, "step", self.global_step)
self._append_step_history(row)
self._record_row(row)
self._step_plateau_scheduler(row)
if self._should_stop_early(improved):
    return
```

When no validation loader exists, `val_every_n_steps` still creates check
boundaries for a training monitor. When `val_every_n_steps is None`, the final
step is the only check. A default `val_loss` monitor with no validation and no
patience returns `None` without failing. A missing custom monitor names the
one-based step in its error.

Add the plateau helper at the same point so Task 5 only has to finish normal
interval dispatch:

```python
def _step_plateau_scheduler(self, row: dict[str, object]) -> None:
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau
    if isinstance(self.scheduler, plateau):
        self.scheduler.step(float(row["val_loss"]))
```

The existing pre-fit guard guarantees a plateau scheduler always has
validation data, and this helper is called only on validation rows.

Narrow the existing validation guard so custom training monitors can work
without a validation loader:

```python
if (
    self.patience is not None
    and self.monitor == "val_loss"
    and self.num_val_batches == 0
):
    raise ValueError("patience needs validation data for monitor 'val_loss'.")
```

- [ ] **Step 4: Add missing/non-finite and no-validation monitor tests**

```python
def test_training_metric_can_monitor_step_run_without_validation():
    trainer = Trainer(
        max_steps=3, val_every_n_steps=1,
        monitor="train_loss", mode="min", device="cpu", plot=False,
    )
    trainer.fit(CountingModel(), TinyData(val_batches=0))
    assert trainer.best_step in {1, 2, 3}


def test_missing_step_monitor_reports_step():
    trainer = Trainer(
        max_steps=2, val_every_n_steps=1,
        monitor="iou", mode="max", device="cpu", plot=False,
    )
    with pytest.raises(ValueError, match="iou.*step 1"):
        trainer.fit(CountingModel(), TinyData())
```

- [ ] **Step 5: Run monitor and all step tests**

Run: `rtk uv run pytest tests/test_step_training.py tests/test_checkpoint.py tests/test_trainer.py -q`

Expected: PASS with epoch and step patience using the same bad-check counter.

- [ ] **Step 6: Commit step monitor integration**

```bash
rtk git add deeptool/trainer.py tests/test_step_training.py
rtk git commit -m "feat: monitor validation by optimizer step"
```

### Task 5: Support Step and Epoch Scheduler Intervals

**Files:**
- Modify: `deeptool/trainer.py`
- Modify: `deeptool/module.py`
- Modify: `tests/test_step_training.py`
- Modify: `tests/test_module.py`

**Interfaces:**
- Consumes: current `(optimizer, scheduler)` configuration and Task 2's update primitive.
- Produces: normal scheduler intervals `step`/`epoch`; validation-driven plateau stepping; unit-aware notebook plot coordinates.

- [ ] **Step 1: Write failing normal-scheduler timing tests**

```python
# tests/test_step_training.py
class ScheduledModel(CountingModel):
    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.1)
        return optim, torch.optim.lr_scheduler.StepLR(
            optim, step_size=1, gamma=0.1
        )


def test_auto_scheduler_steps_after_each_update_in_step_mode(tmp_path):
    trainer = Trainer(
        max_steps=3, device="cpu", plot=False,
        log_dir=tmp_path / "exp",
    )
    trainer.fit(ScheduledModel(), TinyData(val_batches=0))
    rows = _rows(tmp_path / "exp/history.jsonl")

    assert [row["lr"] for row in rows] == pytest.approx([0.1, 0.01, 0.001])


def test_epoch_interval_scheduler_steps_on_loader_exhaustion_in_step_mode():
    trainer = Trainer(
        max_steps=5, scheduler_interval="epoch",
        device="cpu", plot=False,
    )
    trainer.fit(ScheduledModel(), TinyData(train_batches=2, val_batches=0))

    assert trainer.optim.param_groups[0]["lr"] == pytest.approx(0.001)
```

The second test has two completed two-batch passes before stopping midway
through the third, so the scheduler must step exactly twice.

- [ ] **Step 2: Add a failing step-mode ReduceLROnPlateau test**

```python
class PlateauStepModel(CountingModel):
    def __init__(self, losses):
        super().__init__()
        self.losses = iter(losses)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, patience=0, factor=0.1
        )
        return optim, scheduler

    def validation_step(self, batch):
        return torch.tensor(next(self.losses))


def test_plateau_steps_only_after_step_validation(tmp_path):
    trainer = Trainer(
        max_steps=6, val_every_n_steps=2, log_every_n_steps=2,
        device="cpu", plot=False, log_dir=tmp_path / "exp",
    )
    trainer.fit(PlateauStepModel([0.5, 0.7, 0.8]), TinyData())
    rows = _rows(tmp_path / "exp/history.jsonl")

    assert [row["lr"] for row in rows] == pytest.approx([0.1, 0.1, 0.01])
```

The scripted losses rise after the first check, proving plateau scheduling is
driven by validation loss and occurs only after validation rows.

- [ ] **Step 3: Complete scheduler dispatch and preserve LR-used recording**

Resolve `auto` once after optimizer configuration. Call normal step schedulers
after `optimizer.step()`, normal epoch schedulers only after a train loader is
exhausted, and plateau schedulers only after validation with `val_loss`.

Capture `lr_used` before the scheduler call and persist that captured value so
each row describes the final optimizer update in its window. Do not read the
post-scheduler optimizer LR while assembling the row.

- [ ] **Step 4: Make Module plotting use the active progress unit**

Add Trainer helpers:

```python
def plot_x(self, train: bool) -> float:
    if self.training_unit == "step":
        return float(self.global_step + 1 if train else self.global_step)
    if train:
        return self.train_batch_idx / self.num_train_batches
    return float(self.epoch + 1)
```

Use this from `Module.plot()` and set the `ProgressBoard` xlabel to
`training_unit`. Add a `tests/test_module.py` assertion that step-mode board
points use optimizer steps while the existing epoch fractional coordinates
remain unchanged.

- [ ] **Step 5: Run scheduler, board, and epoch regression tests**

Run: `rtk uv run pytest tests/test_step_training.py tests/test_trainer.py tests/test_module.py tests/test_board.py -q`

Expected: PASS, including the existing epoch StepLR and plateau sequences.

- [ ] **Step 6: Commit scheduler and plotting support**

```bash
rtk git add deeptool/trainer.py deeptool/module.py tests/test_step_training.py tests/test_module.py
rtk git commit -m "feat: schedule and plot by training unit"
```

### Task 6: Persist Step-Aware Checkpoints and Matplotlib Plots

**Files:**
- Modify: `deeptool/checkpoint.py`
- Modify: `deeptool/trainer.py`
- Modify: `deeptool/record.py`
- Modify: `tests/test_checkpoint.py`
- Modify: `tests/test_step_training.py`
- Modify: `tests/test_record.py`

**Interfaces:**
- Consumes: step-mode `global_step`, generic snapshot progress key, `RunData`.
- Produces: optional `step` in full checkpoints and metadata returned by `load_checkpoint()`; step-aware `plot_runs()` x axis.

- [ ] **Step 1: Add failing full-checkpoint tests**

```python
# tests/test_step_training.py
def test_step_checkpoint_includes_and_returns_global_step(tmp_path):
    trainer = Trainer(max_steps=3, device="cpu", plot=False)
    trainer.fit(CountingModel(), TinyData(val_batches=0))
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(path)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["step"] == 3
    meta = Trainer.load_checkpoint(path, CountingModel())
    assert meta["step"] == 3
```

- [ ] **Step 2: Extend checkpoint payloads without changing epoch files**

Add a keyword-only optional argument:

```python
def checkpoint_payload(model, optim, epoch: int, *,
                       step: int | None = None) -> dict[str, Any]:
    payload = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "hparams": getattr(model, "hparams", {}),
    }
    if step is not None:
        payload["step"] = step
    return payload
```

Keep standalone `save_checkpoint(model, optim, epoch, path)` unchanged.
`Trainer.save_checkpoint()` supplies `step=self.global_step` only in step mode.
`load_checkpoint()` returns `epoch`, `hparams`, and `step` when present. Do not
implement automatic resume.

- [ ] **Step 3: Add a failing step-axis Matplotlib test**

```python
# tests/test_record.py
def test_plot_runs_uses_step_axis_for_step_records(tmp_path):
    recorder = RunRecorder(tmp_path / "llm")
    recorder.epoch(step=10, train_loss=2.0)
    recorder.epoch(step=20, train_loss=1.5)

    figure = plot_runs(tmp_path)[0]
    axes = figure.axes[0]
    assert axes.get_xlabel() == "step"
    assert list(axes.lines[0].get_xdata()) == [10, 20]
    plt.close("all")
```

- [ ] **Step 4: Treat both progress keys as structural in `plot_runs()`**

Exclude both `epoch` and `step` from metric figures. For each run, select
`epoch` when present, otherwise `step`, otherwise a positional range. Set the
xlabel from the selected structural key. Existing all-epoch plots remain
unchanged; mixed-unit separation belongs to the Streamlit dashboard plan.

- [ ] **Step 5: Run checkpoint, record, and step tests**

Run: `rtk uv run pytest tests/test_checkpoint.py tests/test_record.py tests/test_step_training.py tests/test_trainer.py -q`

Expected: PASS with exact existing epoch checkpoint keys and new step metadata.

- [ ] **Step 6: Commit persistence support**

```bash
rtk git add deeptool/checkpoint.py deeptool/trainer.py deeptool/record.py tests/test_checkpoint.py tests/test_step_training.py tests/test_record.py
rtk git commit -m "feat: persist optimizer-step progress"
```

### Task 7: Document Step Training and Run Full Gates

**Files:**
- Modify: `README.md`
- Modify: `docs/guide/trainer.md`
- Modify: `docs/guide/trainer.en.md`
- Modify: `docs/guide/best.md`
- Modify: `docs/guide/best.en.md`
- Modify: `docs/guide/module.md`
- Modify: `docs/guide/module.en.md`

**Interfaces:**
- Consumes: all finalized step-mode APIs.
- Produces: matching Korean/English guidance and no new runtime interface.

- [ ] **Step 1: Add the canonical step-mode example in both languages**

```python
trainer = dt.Trainer(
    max_steps=10_000,
    log_every_n_steps=50,
    val_every_n_steps=500,
    scheduler_interval="step",
    patience=4,
    monitor="val_loss",
)
trainer.fit(model, data)
```

Explain one-based optimizer steps, finite-loader repetition, final validation,
log windows, validation-check patience, scheduler intervals, `best_step`, and
step checkpoint metadata. State explicitly that AMP, gradient accumulation,
infinite IterableDatasets, and resume are not included.

- [ ] **Step 2: Update scheduler and Module plotting documentation**

Replace the statement that batch schedulers are unsupported with the exact
`scheduler_interval="step"` contract. Keep the separate plateau rule. Explain
that notebook plots use the active epoch/step x coordinate.

- [ ] **Step 3: Update README API and recording examples**

Show one epoch JSONL row and one step JSONL row. Add `best_step` and
`global_step` to the Trainer API description without removing `best_epoch`.

- [ ] **Step 4: Run all automated checks**

Run: `rtk uv run pytest -q`

Expected: PASS.

Run: `rtk uv run --group docs mkdocs build --strict`

Expected: PASS.

- [ ] **Step 5: Commit step-training documentation**

```bash
rtk git add README.md docs/guide/trainer.md docs/guide/trainer.en.md docs/guide/best.md docs/guide/best.en.md docs/guide/module.md docs/guide/module.en.md
rtk git commit -m "docs: explain optimizer-step training"
```
