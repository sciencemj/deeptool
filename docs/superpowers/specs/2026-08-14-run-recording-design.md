# Run Recording and Scheduler Design

## Scope

Add persistent epoch records, custom scalar logging, run loading and plotting,
epoch-based learning-rate scheduling, and one-line headless progress output.
Automatic mixed precision and batch-stepped schedulers are out of scope.

With `log_dir=None`, `Trainer` preserves its current behavior and output. The
headless progress line is enabled only when `log_dir` is set and execution is
outside a notebook.

## Chosen approach

`Trainer` owns an epoch scalar accumulator and an optional `RunRecorder`.
`Module.log()` forwards scalar observations to that accumulator. At the end of
each epoch, `Trainer` averages custom observations, combines them with its
built-in loss, learning-rate, and duration fields, and gives the completed row
to the recorder.

This is preferable to persisting `ProgressBoard.data`: board points are
downsampled for display and do not exist with `plot=False`. It is also
preferable to introducing a generic event bus, which would add indirection for
only one current sink.

`Module.plot()` remains the live-board API. `Module.log()` records an epoch
metric and also plots it when a board exists, inferring the training or
validation phase from `Module.training`. One call therefore serves both
persistent and notebook views without changing existing `plot()` behavior.
In a plain script, `Trainer(plot=True)` creates a board with display disabled:
board data still accumulates, but no unused Matplotlib figures are created.

## Public API

### Recording

```python
trainer = deeptool.Trainer(max_epochs=50, log_dir="runs/exp1")

class Model(deeptool.Module):
    def validation_step(self, batch):
        loss = ...
        self.log("iou", iou)
        return loss
```

`Trainer.__init__` accepts `log_dir: str | Path | None = None`.

`Module.log(key, value)` accepts a scalar tensor or float. The recorder stores
`key` exactly as supplied and never interprets or prefixes it. Multiple values
with the same name during one epoch are averaged. When a live board exists,
`log()` also delegates to `plot()`; only that visual label receives the board's
existing `train_` or `val_` prefix. If both phases need separate persisted
metrics, the model names them explicitly, for example `train_iou` and
`val_iou`.

Custom epoch averages are also appended to `Trainer.history`, regardless of
whether `log_dir` is set. Calling `log()` before a Trainer is attached remains
a no-op, matching `plot()`.

### Recorder

```python
class RunRecorder:
    def __init__(self, log_dir: str | Path): ...
    def meta(self, **info: object) -> None: ...
    def epoch(self, **scalars: object) -> None: ...
```

`RunRecorder` creates `log_dir`. `meta()` writes `meta.json` atomically so a
partially written metadata document is not exposed. `epoch()` appends one JSON
object plus a newline to `history.jsonl`, flushes it immediately, and closes the
file after the call. Reusing an existing directory appends history rather than
discarding it.

Torch scalar tensors are converted to Python numbers. Non-scalar tensors and
non-numeric custom metrics raise `ValueError` at the `Module.log()` call so an
invalid metric cannot silently corrupt a long-running experiment.

### Reading and plotting

```python
runs = deeptool.load_runs("runs")
figures = deeptool.plot_runs("runs")
```

`load_runs(root)` reads each immediate child directory containing
`history.jsonl` and returns:

```python
{
    "exp1": {
        "epoch": [0, 1],
        "train_loss": [2.47, 2.20],
        "iou": [0.32, 0.36],
    }
}
```

Metric lists align with the epoch list. A key absent from an epoch is padded
with `None`, allowing `plot_runs()` to preserve the correct x coordinate and
show a gap. Empty lines are ignored. A malformed non-empty JSONL line raises
`ValueError` containing the file and line number.

`plot_runs(root)` accepts the same path or an already-loaded mapping. It creates
one Matplotlib figure per metric other than `epoch`, overlays every run that has
that metric, uses that run's recorded epoch values for x coordinates, and
returns the figures without calling `show()`. Missing metrics therefore work
naturally across heterogeneous models. No dependency is added.

## Metadata and epoch schema

At the start of `fit()`, `meta.json` contains:

- `started_at`: timezone-aware ISO 8601 timestamp
- `device`: resolved device string
- `model_class`: fully qualified model class name
- `trainer_hparams`: JSON-safe Trainer hyperparameters
- `model_hparams`: JSON-safe model hyperparameters

Path objects, devices, and other metadata values not directly supported by
JSON are represented with `str(value)` rather than preventing training.

Each completed epoch row contains:

- `epoch`: zero-based epoch index
- `train_loss`
- `val_loss` when validation data exists
- all custom metrics observed in that epoch
- `lr`: learning rate used during that epoch from the optimizer's first
  parameter group
- `sec`: elapsed wall-clock seconds for the epoch

The row is persisted before early-stopping evaluation. If the process stops
later, all already completed rows remain readable.

## Scheduler integration

`Module.configure_optimizers()` may return either an optimizer, preserving the
existing contract, or `(optimizer, scheduler)`.

After the epoch row is assembled and recorded:

- `torch.optim.lr_scheduler.ReduceLROnPlateau` receives the current
  `val_loss` through `scheduler.step(val_loss)`.
- Other epoch-based schedulers receive `scheduler.step()`.

`ReduceLROnPlateau` without validation data raises `ValueError` before training
starts. The recorded `lr` is the learning rate used for the row's epoch; a
scheduler change applies to the next epoch. Per-batch schedulers such as
`OneCycleLR` are not supported by this contract and are documented as out of
scope.

## Headless output

When `log_dir` is set and execution is outside an IPython kernel, `Trainer`
prints one line after recording each completed epoch. Fields follow the JSONL
row order and use a stable compact representation, for example:

```text
epoch=0 train_loss=2.4724 val_loss=2.3155 iou=0.3248 lr=0.001 sec=116.2
```

No line is printed in notebooks or when `log_dir=None`, preserving existing
output behavior.

## Failure behavior

Creating or writing the requested run directory is part of the training
contract: filesystem errors propagate immediately. Loading a missing root
raises `FileNotFoundError`; a root with no histories returns an empty mapping.
Recording does not swallow serialization or I/O failures.

## Verification

Tests will cover:

1. Existing behavior and silence when `log_dir=None`.
2. Metadata creation and one immediately readable JSONL row per epoch with
   `plot=False`.
3. Preservation of completed rows when training raises during a later epoch.
4. Averaging arbitrary custom metrics from tensors and floats while preserving
   their persisted names, plus train/validation prefixes only on board labels.
5. Loading and plotting runs with disjoint metric sets.
6. Clear malformed-JSONL errors.
7. Scheduler-free backward compatibility, normal epoch scheduler stepping, and
   validation-loss-driven `ReduceLROnPlateau` stepping.
8. Headless output only for logged, non-notebook runs.
