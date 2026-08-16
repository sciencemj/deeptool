# Configurable Best-Model Monitor Design

## Scope

Allow best-model snapshots and early stopping to use a logged scalar such as
IoU or accuracy instead of always using validation loss. One monitor and one
optimization direction jointly control snapshot selection, `restore_best()`,
and early stopping.

Threshold stopping, multiple simultaneous monitors, smoothed metrics, and
different criteria for snapshotting and early stopping are out of scope.
`ReduceLROnPlateau` remains driven by `val_loss` and is not coupled to this
monitor.

## Public API

`Trainer` adds two constructor arguments:

```python
trainer = deeptool.Trainer(
    max_epochs=100,
    patience=5,
    monitor="iou",
    mode="max",
)
```

- `monitor: str = "val_loss"` is the exact row key to compare.
- `mode: Literal["min", "max"] = "min"` selects whether lower or higher is
  better.

The defaults exactly preserve validation-loss behavior. Metric names are not
guessed. A validation IoU can be supplied with `self.log("iou", value)`. If a
model records both phases, it names them explicitly, for example
`train_iou` and `val_iou`, and monitors `val_iou`.

New state exposed by `Trainer`:

- `best_score`: best finite value of the configured monitor;
- `best_epoch`: zero-based best epoch in epoch mode, otherwise `None`;
- `best_step`: one-based best optimizer step in step mode, otherwise `None`.

`best_val_loss` continues to mean the minimum validation loss observed. With
the default monitor it equals `best_score` and corresponds to `best_epoch`.
With a custom monitor it remains useful diagnostic information but need not
come from the custom monitor's best epoch or step.

`restore_best()` continues returning an integer for compatibility. Its unit is
the active training unit: epoch in epoch mode and optimizer step in step mode.
The `best_epoch` and `best_step` properties remove ambiguity for inspection.

## Metric resolution and update order

At each monitor boundary, `Trainer` first finalizes built-in losses and averages
all custom scalar observations. It assembles the same scalar row used for
history and recording, resolves `monitor` by exact key, validates the value,
and then updates the generic best snapshot.

In epoch mode the monitor boundary is the end of every epoch. In step mode it
is each `val_every_n_steps` boundary and the final step, whether or not a
validation loader exists. When validation data exists, validation runs before
the monitor is resolved. Step-mode patience therefore counts monitor checks,
not raw optimizer steps.

The completed row is persisted before the early-stopping decision so an epoch
or step that triggers stopping remains visible. A normal scheduler keeps its
configured interval. `ReduceLROnPlateau` separately receives `val_loss` after a
validation boundary, preserving the previously approved contract.

When the monitor improves, one update records the score, progress unit and
value, and optional model snapshot. `mode="min"` uses strict decrease;
`mode="max"` uses strict increase. Equal values are not improvements. No
`min_delta` tolerance is added.

## Snapshot representation and compatibility

`BestSnapshot` becomes metric-neutral internally. A saved best snapshot records
the configured monitor name, mode, best score, and either `epoch` or `step`
beside the model state. When the monitor is `val_loss`, it also retains the
existing `val_loss` field so previously documented inspection continues to
work.

Restoring existing best files remains supported because restore only requires
their model state. Existing default-mode files and calls do not change. Error
messages use the configured metric and score instead of incorrectly calling a
custom value validation loss.

`snapshot_best=False` still tracks the best score and progress, but
`restore_best()` reports that no snapshot was retained.

## Validation and failure behavior

- `mode` must be exactly `min` or `max`.
- `monitor` must be a non-empty string.
- A custom monitored key must exist at every applicable monitor boundary.
- The monitored value must be a scalar finite number; NaN and infinity fail
  immediately.
- `monitor="val_loss"` with no validation data and no patience preserves the
  current behavior: training runs, no best point is selected, and
  `restore_best()` remains unavailable. Adding patience or
  `ReduceLROnPlateau` still requires validation data.
- A custom training metric is technically supported because names are free
  form, although documentation recommends validation metrics for model
  selection.

Missing or invalid monitors are errors rather than non-improvements. Silently
spending patience on absent data could stop a long run for the wrong reason.

## Verification and documentation

Tests cover:

1. Unchanged default minimum-`val_loss` selection and early stopping.
2. Maximum IoU and accuracy selection through `Module.log()`.
3. The same custom monitor controlling `best.pt`, `best_score`, best progress,
   early stopping, and `restore_best()`.
4. Equal values not counting as improvements in either mode.
5. Invalid mode, empty monitor, missing metric, NaN, and infinity errors.
6. `snapshot_best=False` tracking without restoration.
7. `best_val_loss` remaining the true loss minimum under a custom monitor.
8. `ReduceLROnPlateau` continuing to receive validation loss rather than the
   custom monitor.
9. Generic snapshot metadata and restoration of the prior snapshot format.
10. Epoch-mode `best_epoch` and step-mode `best_step` semantics.

Korean and English best-model guides show loss-minimizing and IoU-maximizing
examples and state clearly that snapshotting and early stopping share the
monitor.
