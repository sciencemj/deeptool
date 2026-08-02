# Quickstart

One linear regression, start to finish, showing the five pieces of `deeptool`.
To run the same thing as a notebook, open
[`examples/quickstart.ipynb`](https://github.com/sciencemj/deeptool/blob/main/examples/quickstart.ipynb).

```python
import torch
from torch import nn
from torch.nn import functional as F

import deeptool as dt
```

## 1. Data

Subclass `DataModule` and implement `get_dataloader(train)`. That is the only
required method. Split training from validation with an index slice.

```python
class SyntheticRegression(dt.DataModule):
    def __init__(self, n=200, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(0)
        self.X = torch.randn(n, 2)
        self.y = self.X @ torch.tensor([[2.0], [-3.4]]) + 4.2

    def get_dataloader(self, train):
        idx = slice(0, 160) if train else slice(160, None)
        return self.get_tensorloader((self.X, self.y), train, idx)


data = SyntheticRegression()
data.hparams
```

```
{'n': 200, 'batch_size': 32}
```

`save_hyperparameters()` turned the `__init__` arguments into attributes and
collected them in `hparams`. That is why `get_tensorloader` can read
`self.batch_size` even though you never assigned it.

## 2. Model

Subclass `Module` and assign `self.net`. `forward` is delegated automatically.

`nn.LazyLinear` lets you skip the input size — it is settled on the first
forward pass.

```python
class LinearRegression(dt.Module):
    def __init__(self, lr=0.03):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
```

## 3. Attaching methods from a later cell

This is the heart of notebook work. If you forgot `loss` and
`configure_optimizers`, **you do not go back and re-run the class cell.**

```python
@dt.add_to_class(LinearRegression)
def loss(self, y_hat, y):
    return F.mse_loss(y_hat, y)


@dt.add_to_class(LinearRegression)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

Instances you already created pick the methods up immediately, because they
land on the class.

## 4. Training

```python
model = LinearRegression()
trainer = dt.Trainer(max_epochs=20)
trainer.fit(model, data)
```

The loss curve **updates in place** in this cell's output. One figure keeps
being redrawn instead of a new one piling up each epoch.

The device is chosen for you: `mps` on a Mac, `cuda` with an NVIDIA GPU.

```python
trainer.device
```

```
device(type='mps')
```

You used `nn.LazyLinear` without initializing it by hand. `fit()` runs one
dummy forward pass before building the optimizer, which materializes the
parameters.

Per-epoch losses land in `history`.

```python
trainer.history["train_loss"][-1], trainer.history["val_loss"][-1]
```

```
(0.00033, 0.00041)
```

## 5. Evaluation and checkpoints

Run the trained model over the whole validation set.

```python
p = trainer.predict(data)
p.outputs.shape, len(p)
```

```
(torch.Size([40, 1]), 40)
```

For a classifier you would read `p.accuracy`, `p.preds` and `p.confidence`
straight off. Regression uses `p.outputs` directly.

Save and restore:

```python
trainer.save_checkpoint("linreg.pt")

restored = LinearRegression()
restored(data.X[:1])  # materialize LazyLinear
meta = dt.Trainer.load_checkpoint("linreg.pt", restored)
meta
```

```
{'epoch': 19, 'hparams': {'lr': 0.03}}
```

The dummy forward before restoring is required. Until `LazyLinear` has
parameters, `load_state_dict` has nowhere to put the weights.

## Next

- Real datasets: [Data](guide/data.md)
- Keeping the best weights before overfitting: [Best weights & early stopping](guide/best.md)
- Digging into predictions: [Evaluation](guide/evaluate.md)
