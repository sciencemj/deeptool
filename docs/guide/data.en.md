# Data

`dt.DataModule` keeps the training and validation dataloaders on one object.
Subclasses implement **`get_dataloader(train)` and nothing else**.

```python
class MyData(dt.DataModule):
    def get_dataloader(self, train):
        ...
```

`train_dataloader()` and `val_dataloader()` are thin wrappers that call it with
`train=True` and `train=False`.

## Starting from tensors

If the data is already in memory, `get_tensorloader` assembles the
`TensorDataset` and `DataLoader` for you. It shuffles when `train` is true.

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
```

The third argument is the slice that separates training from validation. It is
applied to every tensor, so X and y can never drift apart.

## Call order

!!! warning "`super().__init__()` first, `save_hyperparameters()` second"
    `DataModule.__init__` puts its own `root`, `num_workers` and `batch_size`
    defaults into `hparams`. Reverse the order and the `batch_size` you passed
    is overwritten by the parent's 32.

    ```python
    def __init__(self, batch_size=256):
        super().__init__()            # batch_size=32 (parent default)
        self.save_hyperparameters()   # overwritten with 256
    ```

## Using torchvision datasets

`torchvision` is not a dependency of `deeptool`. Bundling datasets is out of
scope, so install it only if you need it.

```bash
uv run --with torchvision jupyter lab
```

`get_tensorloader` is for tensors, so it does not apply here. Wrap the
torchvision `Dataset` in a `DataLoader` directly.

```python
import torchvision
from torch.utils import data
from torchvision import transforms


class FashionMNIST(dt.DataModule):
    def __init__(self, batch_size=256, root="./data"):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.ToTensor()
        self.train_ds = torchvision.datasets.FashionMNIST(
            root=root, train=True, transform=trans, download=True)
        self.val_ds = torchvision.datasets.FashionMNIST(
            root=root, train=False, transform=trans, download=True)

    def get_dataloader(self, train):
        ds = self.train_ds if train else self.val_ds
        return data.DataLoader(ds, self.batch_size, shuffle=train,
                               num_workers=self.num_workers)
```

Check it:

```python
d = FashionMNIST()
X, y = next(iter(d.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
print(len(d.train_dataloader()), len(d.val_dataloader()))
```

```
torch.Size([256, 1, 28, 28]) torch.float32 torch.Size([256]) torch.int64
235 40
```

## Two things that bite

**Image batches are `(N, 1, 28, 28)`.** They are four-dimensional, so the model
needs `nn.Flatten()` as its first layer. Without it you get a shape error.

```python
self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(256),
                         nn.ReLU(), nn.LazyLinear(10))
```

**Why `num_workers` defaults to 0.** Anything above zero can hang in a notebook
on macOS, because of how processes are started there. Raise it only when
running as a script.

## When there is no validation data

Return `None` from `get_dataloader(train=False)` and validation is skipped.

```python
def get_dataloader(self, train):
    if not train:
        return None
    return self.get_tensorloader((self.X, self.y), train)
```

`history["val_loss"]` then stays empty, and both
[`restore_best()`](best.md) and `patience` become unavailable — they are both
keyed on validation loss.

## Next

- [Model](module.md) — what receives these batches
- [Trainer](trainer.md) — when and how the dataloaders are called
