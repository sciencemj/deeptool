# deeptool

A thin helper library for working with PyTorch models object-oriented style in
Jupyter notebooks.

You write the model in plain PyTorch. `deeptool` handles what surrounds it.

```bash
pip install deeptool
```

## Thirty-second example

```python
import torch
from torch import nn
from torch.nn import functional as F

import deeptool as dt


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


class LinearRegression(dt.Module):
    def __init__(self, lr=0.03):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
```

Attach the remaining methods from a later cell. No need to redefine the class.

```python
@dt.add_to_class(LinearRegression)
def loss(self, y_hat, y):
    return F.mse_loss(y_hat, y)


@dt.add_to_class(LinearRegression)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

Train, and the loss curve updates live in the cell output.

```python
trainer = dt.Trainer(max_epochs=20)
trainer.fit(LinearRegression(), SyntheticRegression())
```

## What it covers

| | |
|---|---|
| Hyperparameter capture | `save_hyperparameters()` turns `__init__` arguments into attributes and `hparams` |
| Cross-cell methods | `@add_to_class` without redefining the class |
| Live loss curves | Redrawn in place during training |
| Device selection | `cuda` → `mps` → `cpu` |
| Best weights, early stopping | Snapshot at the lowest validation loss, plus `patience` |
| Post-hoc evaluation | `trainer.predict(data)` collects per-sample predictions |
| Checkpoints | Save and restore |

## What it does not

No model zoo, no bundled datasets, no distributed training, no image
visualization, no logging-backend integrations.

Runtime dependencies are `torch`, `matplotlib` and `ipython`. That list does
not grow.

## Where to start

- [Quickstart](quickstart.md) — a full linear regression run
- [Model](guide/module.md) — `Module` and `add_to_class`
- [Data](guide/data.md) — `DataModule`
- [Trainer](guide/trainer.md) — `Trainer`, devices, `history`
- [Best weights & early stopping](guide/best.md) — `restore_best`, `patience`
- [Evaluation](guide/evaluate.md) — `predict`, `Predictions`
- [API](api.md) — full reference

## License

MIT. The design follows `d2l/torch.py` from
[d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en).
