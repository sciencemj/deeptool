# Model

`dt.Module` is `nn.Module` with hyperparameter capture and plotting hooks on
top. How you write PyTorch models does not change.

## The three things you fill in

```python
class MyNet(dt.Module):
    def forward(self, X): ...
    def loss(self, y_hat, y): ...
    def configure_optimizers(self): ...
```

`loss` and `configure_optimizers` raise `NotImplementedError` by default. You
have to supply them.

`forward` is the exception: **assign `self.net` and it is delegated for you.**

```python
class MyNet(dt.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(256),
                                 nn.ReLU(), nn.LazyLinear(10))
```

With neither `self.net` nor `forward`, the call site stops you.

```
AssertionError: implement forward() or assign self.net
```

## `save_hyperparameters()` and call order

It turns every `__init__` argument into an instance attribute and collects them
in `self.hparams`.

```python
class MyNet(dt.Module):
    def __init__(self, lr=0.01, num_hiddens=256):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(10))

model = MyNet(lr=0.1)
model.lr, model.hparams
```

```
(0.1, {'lr': 0.1, 'num_hiddens': 256})
```

`hparams` is stored in checkpoints, so a file alone tells you what settings the
model was trained with.

!!! warning "The order matters"
    `super().__init__()` **first**, `save_hyperparameters()` **second**.

    The parent `__init__` also overwrites `hparams` with its own arguments.
    Reverse the order and your values are erased by the parent's defaults.

    ```python
    def __init__(self, lr=0.1):
        self.save_hyperparameters()   # stores lr=0.1
        super().__init__()            # parent overwrites hparams
    ```

Local variables are not picked up. Only declared arguments are read.

```python
def __init__(self, lr=0.1):
    scratch = lr * 2      # never reaches hparams
    self.save_hyperparameters()
```

To leave an argument out, use `ignore`.

```python
self.save_hyperparameters(ignore=["api_key"])
```

## `@add_to_class` — attaching methods across cells

It removes the notebook problem where changing a class means re-running its
definition cell, which then forces you to re-run everything below it.

```python
# cell 3
class MyNet(dt.Module):
    def __init__(self, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(10)

# cell 7 — much later
@dt.add_to_class(MyNet)
def loss(self, y_hat, y):
    return F.cross_entropy(y_hat, y)
```

Because it lands on the class, **instances you already built pick it up
immediately.**

The decorator returns the original function, so the name stays usable in the
cell that defined it.

```python
@dt.add_to_class(MyNet)
def loss(self, y_hat, y):
    return F.cross_entropy(y_hat, y)

loss          # <function loss at 0x...> — not None
```

## Batch convention

The default `training_step` and `validation_step` read a batch this way:

```
batch[:-1]   inputs  (there can be several)
batch[-1]    targets
```

Forward is called as `self(*batch[:-1])`, so a model with two inputs takes an
`(X1, X2, y)` batch as is.

## Adding an accuracy curve

The default only plots loss. To see accuracy as well, override
`validation_step`.

```python
@dt.add_to_class(MyNet)
def validation_step(self, batch):
    y_hat = self(*batch[:-1])
    loss = self.loss(y_hat, batch[-1])
    self.plot('loss', loss, train=False)
    self.plot('acc', (y_hat.argmax(-1) == batch[-1]).float().mean(), train=False)
    return loss
```

Scalars sent to `plot` share the board with a different color and line style.
The return value must still be the loss — `Trainer` uses it to fill `history`
and to decide the best epoch.

!!! note "`plot` does not reach `history`"
    `plot` only draws. `trainer.history` accumulates `train_loss` and
    `val_loss` only. To keep accuracy as numbers, collect it yourself, or use
    [`trainer.predict(data).accuracy`](evaluate.md) after training.

## Next

- [Data](data.md) — what feeds batches to the model
- [Trainer](trainer.md) — how `fit` uses these conventions
