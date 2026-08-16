# Trainer

All configuration lives on the `Trainer()` constructor. `fit()` takes two
arguments.

```python
trainer = dt.Trainer(max_epochs=20)
trainer.fit(model, data)
```

One trainer means **one training setup**, which is why `trainer.hparams`
records the whole thing.

```python
dt.Trainer(max_epochs=20, patience=3).hparams
```

```
{'max_epochs': 20, 'device': None, 'gradient_clip_val': 0, 'plot': True,
 'snapshot_best': True, 'best_path': None, 'best_with_optim': False,
 'patience': 3, 'log_dir': None, 'monitor': 'val_loss', 'mode': 'min',
 'max_steps': None, 'log_every_n_steps': 1, 'val_every_n_steps': None,
 'scheduler_interval': 'auto'}
```

## Device selection

The first available of `cuda`, `mps`, `cpu` is chosen. The training loop moves
batches for you.

There are three ways to check, and they mean different things.

```python
dt.default_device()              # what would be chosen (no Trainer needed)
trainer.device                   # what this trainer actually uses
next(model.parameters()).device  # where the model really ended up
```

To force it:

```python
dt.Trainer(max_epochs=20, device="cpu")
```

!!! warning "`hparams['device']` is not the resolved device"
    ```python
    trainer = dt.Trainer(max_epochs=20)
    trainer.hparams['device']   # None
    trainer.device              # device(type='mps')
    ```

    `hparams` stores the **argument you passed**. Pass nothing and it is
    `None`. For the device in use, read `trainer.device`.

## `history`

Mean loss per epoch.

```python
trainer.history
```

```
{'train_loss': [0.62, 0.48, 0.41, ...], 'val_loss': [0.58, 0.45, 0.43, ...]}
```

Its length tells you whether early stopping fired.

```python
len(trainer.history["train_loss"]) < trainer.max_epochs   # True means it stopped early
```

Without validation data, `val_loss` stays an empty list.

When a model calls `self.log("iou", value)`, the epoch average of that custom
metric enters the same dict. An epoch with no observation gets `None` to keep
positions aligned.
`Trainer(monitor="iou", mode="max")` also uses that metric for best snapshots
and early stopping. The defaults are `monitor="val_loss"` and `mode="min"`.

## Step-based training

For workloads such as LLM training, bound the run by optimizer updates with
`max_steps`. Supply exactly one of `max_epochs` and `max_steps`.

```python
trainer = dt.Trainer(
    max_steps=10_000,
    log_every_n_steps=50,
    val_every_n_steps=500,
    scheduler_interval="step",
    patience=4,
    monitor="val_loss",
    log_dir="runs/llm-1",
)
trainer.fit(model, data)
```

A step is one completed `optimizer.step()` and is recorded from 1. The train
DataLoader must be finite and non-empty; it is restarted after each completed
pass. Records are emitted every `log_every_n_steps`, and validation runs every
`val_every_n_steps`. If the validation interval is omitted, validation runs
only at the final step. Patience counts actual monitor checks, not optimizer
updates.

Step history includes a `step` list and aligns `val_loss` with `None` on rows
without validation. Read the best point through `best_step`; epoch runs retain
`best_epoch`.

## Persisting training runs

```python
trainer = dt.Trainer(max_epochs=50, plot=False, log_dir="runs/exp1")
trainer.fit(model, data)
```

Every completed epoch or configured step boundary appends and immediately
flushes one JSONL line.

```
runs/exp1/
  meta.json
  history.jsonl
```

`meta.json` contains the start time, resolved device, model class, and Trainer
and model hyperparameters. Each `history.jsonl` row has a free-form schema:

```json
{"epoch": 0, "train_loss": 2.4724, "val_loss": 2.3155, "iou": 0.3248, "lr": 0.001, "sec": 116.2}
```

Step rows use `step` instead of `epoch`. If the process dies during the next
interval, all completed lines remain. A plain script also prints these fields
at each record boundary, but only when `log_dir` is set.
Notebooks retain the live board without those lines. With `log_dir=None`, both
files and epoch output remain disabled.

Load several runs and overlay them one figure per metric:

```python
runs = dt.load_runs("runs")
figures = dt.plot_runs(runs)
```

`load_runs` returns `{run_name: {metric_name: [values, ...]}}`. Metrics present
in only some models need no shared schema; a value absent from an intermediate
progress point is aligned with `None`. `plot_runs` returns one open Matplotlib
Figure for every metric except `epoch` and `step`, using the run's progress
axis.

## Learning-rate schedulers

`configure_optimizers()` may return the existing optimizer-only result or an
`(optimizer, scheduler)` pair.

```python
def configure_optimizers(self):
    optim = torch.optim.Adam(self.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=10, gamma=0.1
    )
    return optim, scheduler
```

With `scheduler_interval="auto"`, a normal scheduler follows the training unit:
once after each epoch in epoch mode, or after every optimizer update in step
mode. Set `"epoch"` or `"step"` explicitly to override that choice. A row's
`lr` is the value used by its final update; a changed value appears on the next
update.

`ReduceLROnPlateau` ignores the normal interval and receives
`scheduler.step(val_loss)` whenever validation runs. Because it needs that
measurement, using it without a validation dataloader raises `ValueError`
before training. AMP is not supported yet.

## Automatic lazy materialization

`nn.LazyLinear` and `nn.LazyConv2d` have no parameters until the first forward
pass. Calling `configure_optimizers()` before that fails.

`fit()` runs one dummy forward under `torch.no_grad()` **before** building the
optimizer, so no manual initialization is needed.

```python
class MyNet(dt.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.LazyLinear(10)   # input size left out

trainer.fit(MyNet(), data)             # just works
```

!!! note "Restoring a checkpoint is the exception"
    `load_checkpoint` does not go through `fit()`, so run one forward pass
    yourself.

    ```python
    restored = MyNet()
    restored(data.X[:1])               # parameters appear here
    dt.Trainer.load_checkpoint("ckpt.pt", restored)
    ```

## Gradient clipping

```python
dt.Trainer(max_epochs=20, gradient_clip_val=1.0)
```

Applies `clip_grad_norm_` after `backward()` and before `optim.step()`. Zero,
the default, does nothing.

## Checkpoints

```python
trainer.save_checkpoint("ckpt.pt")
```

The file holds `model`, `optim`, `epoch`, and `hparams`. Step runs also include
the current `step`, which `load_checkpoint()` returns in its metadata. Loading
does not automatically continue from that step.

Restoring is a static method, so no trainer is needed.

```python
model = MyNet()
model(data.X[:1])                                    # materialize LazyLinear

# for inference — weights only
meta = dt.Trainer.load_checkpoint("ckpt.pt", model)

# to resume training — optimizer state too
optim = model.configure_optimizers()
meta = dt.Trainer.load_checkpoint("ckpt.pt", model, optim)

meta
```

```
{'epoch': 19, 'hparams': {'lr': 0.03}}
```

Passing `optim` is what separates the two uses.

!!! danger "Only load files you trust"
    `hparams` can hold arbitrary Python objects, so the file is read with
    `weights_only=False`. Do not open checkpoints of unknown origin.

## Inside the training loop

What `fit()` does, in order:

1. Takes both dataloaders from `data` and counts the batches
2. Rejects `patience` without validation data, right here
3. Injects `model.trainer` and `model.board`
4. Materializes lazy parameters with a dummy forward
5. Builds the optimizer and optional scheduler via `configure_optimizers()`
6. Writes run metadata when recording is enabled
7. Epoch loop — train, validate, update best, record, step scheduler, stop early

One epoch (`fit_epoch`) walks the training batches calling `training_step`,
then the validation batches calling `validation_step` under `torch.no_grad()`.
Switching between `model.train()` and `model.eval()` happens there too.

## Next

- [Best weights & early stopping](best.md) — step 7's best-epoch logic
- [Evaluation](evaluate.md) — after training finishes
