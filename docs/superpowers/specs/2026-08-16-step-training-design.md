# Step-Based Training Design

## Scope

Add a true optimizer-step training mode for workloads such as language-model
training where a complete dataset epoch is too large to be the useful control
unit. Step mode provides bounded optimizer updates, interval logging,
interval validation, step-based scheduler operation, persistent progress, and
dashboard compatibility.

The implementation repeats a finite, non-empty train DataLoader. Infinite or
lengthless IterableDatasets, gradient accumulation, automatic mixed precision,
distributed training, token-count scheduling, and resume-from-step behavior
are out of scope.

## Public API and mode selection

`Trainer` accepts:

```python
trainer = deeptool.Trainer(
    max_steps=10_000,
    log_every_n_steps=50,
    val_every_n_steps=500,
    scheduler_interval="step",
)
```

Constructor additions are:

- `max_steps: int | None = None`;
- `log_every_n_steps: int = 1`;
- `val_every_n_steps: int | None = None`;
- `scheduler_interval: Literal["auto", "epoch", "step"] = "auto"`.

`max_epochs` becomes optional solely so `Trainer(max_steps=...)` is possible.
Exactly one of `max_epochs` and `max_steps` must be a positive integer. Existing
positional and keyword `max_epochs` calls retain their behavior.

`scheduler_interval="auto"` resolves to epoch in epoch mode and step in step
mode. An explicit value overrides it for normal schedulers. A step is one
completed `optimizer.step()`; without gradient accumulation, this is currently
one train batch. Persisted steps are one-based.

## Training loop structure

The shared batch operation prepares a batch, computes loss, clears gradients,
backpropagates, clips if configured, updates the optimizer, advances the global
step, and optionally advances a step scheduler. Epoch mode composes that
operation over one DataLoader pass and retains its existing zero-based epoch
history.

Step mode repeatedly obtains batches from the finite train DataLoader. At
exhaustion it creates a fresh iterator and continues until `max_steps`, allowing
an epoch-interval scheduler to advance at each complete pass if explicitly
requested. An empty loader fails before entering the loop.

Train loss and custom scalar observations accumulate from the previous emitted
row. A row is emitted when any of these is true:

- `global_step` is divisible by `log_every_n_steps`;
- validation is due;
- `global_step == max_steps`.

This ensures validation results and final partial logging windows are always
persisted. If log and validation boundaries coincide, only one row is emitted.
The row's `train_loss` and `sec` describe the window since the previous row,
not the entire run.

## Validation, monitor, and early stopping

With validation data, `val_every_n_steps=N` runs the full validation DataLoader
after steps `N`, `2N`, and so on. Validation also runs at `max_steps` when that
step was not already a boundary. With `val_every_n_steps=None`, validation runs
only at `max_steps`.

Each validation boundary finalizes validation loss and custom metrics, updates
the configured best-model monitor, records the row, advances
`ReduceLROnPlateau` with `val_loss`, and evaluates early stopping. `patience` in
step mode counts consecutive validation boundaries without monitor
improvement. With only final validation, patience cannot shorten the run; users
who want early stopping must choose a validation interval.

If no validation loader exists, `val_every_n_steps` still defines monitor-check
and row boundaries; with `None`, the only monitor check is the final step. A
training metric may be monitored at those boundaries, but `val_loss` and
validation-only monitors remain unavailable. The default `val_loss` monitor
without patience preserves existing no-validation behavior by selecting no
best point rather than failing.

The generic best state exposes `best_step`; `best_epoch` is `None` in step mode.
`restore_best()` restores the model from the best monitored step and returns
that one-based step.

## Scheduler behavior

Normal schedulers support two intervals:

- `step`: call `scheduler.step()` after every optimizer update, so the new
  learning rate applies to the next update;
- `epoch`: call it after every completed train DataLoader pass.

This applies in either training mode, enabling batch schedulers such as
`OneCycleLR` without changing `Module.configure_optimizers()` beyond its
existing `(optimizer, scheduler)` return value.

`ReduceLROnPlateau` is metric-driven rather than interval-driven. It ignores
`scheduler_interval`, requires validation data, and receives `val_loss` after
each validation execution. In epoch mode this remains once per epoch; in step
mode it follows `val_every_n_steps` plus the final validation.

The row records the learning rate used for its final optimizer update. A
scheduler change affects the following update and is therefore visible in the
next row.

## History, records, plotting, and checkpoints

Step-mode JSONL rows use `step` instead of `epoch`:

```json
{"step": 500, "train_loss": 2.1, "val_loss": 2.3, "lr": 0.0003, "sec": 41.2}
```

Fields that are not observed at a row are aligned with `None`, preserving the
free-form run schema. Step-mode `Trainer.history` includes a `step` list so
metrics from irregular log and validation boundaries retain their x
coordinates; epoch-mode history is unchanged. Headless output prints one line
per emitted row using the same field order. `meta.json` already captures
Trainer hyperparameters and therefore records the selected unit, limits, and
intervals.

`load_runs()` accepts either structural x key. `plot_runs()` uses `step` when
`epoch` is absent. The Streamlit dashboard separates epoch and step chart
groups instead of overlaying incompatible units.

Best snapshots store `step` for step runs. Trainer-created full checkpoints
include the current global step alongside existing payload fields, and
`load_checkpoint()` returns it when present; existing epoch checkpoint loading
remains compatible. Automatic continuation from a stored step is a separate
future feature.

## Validation and failure behavior

- Both limits set, neither limit set, non-integer booleans, and non-positive
  limits are rejected.
- `log_every_n_steps` must be a positive integer.
- `val_every_n_steps` is `None` or a positive integer.
- `scheduler_interval` must be `auto`, `epoch`, or `step`.
- A lengthless or empty train loader produces an actionable error before
  training.
- `ReduceLROnPlateau` without validation data remains an error.
- Monitor failures follow the configurable-monitor design and include the
  failing step.

## Verification and documentation

Tests cover:

1. Every existing epoch-mode test without behavior changes.
2. Exact optimizer update count and one-based global steps.
3. DataLoader repetition and rejection of empty or lengthless loaders.
4. Logging windows, validation-only rows, coincident boundaries, and the final
   partial window.
5. Final-only validation when no interval is supplied.
6. Periodic validation and patience counted in validation checks.
7. Custom monitor selection, `best_step`, snapshots, and restoration.
8. Normal scheduler calls at step and epoch intervals in both modes.
9. Validation-driven `ReduceLROnPlateau` calls in step mode.
10. Step JSONL, headless output, `load_runs()`, `plot_runs()`, and separate
    dashboard axis groups.
11. Constructor and loader failure cases.

Korean and English Trainer guides explain the two exclusive training modes,
the meaning of optimizer step, validation cadence, scheduler intervals, and
the explicit exclusions of AMP and gradient accumulation.
