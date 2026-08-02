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
 'patience': 3}
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

The file holds four things: `model`, `optim`, `epoch`, `hparams`.

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
5. Builds the optimizer via `model.configure_optimizers()`
6. Epoch loop — train, validate, check for a new best, check early stopping

One epoch (`fit_epoch`) walks the training batches calling `training_step`,
then the validation batches calling `validation_step` under `torch.no_grad()`.
Switching between `model.train()` and `model.eval()` happens there too.

## Next

- [Best weights & early stopping](best.md) — step 6's best-epoch logic
- [Evaluation](evaluate.md) — after training finishes
