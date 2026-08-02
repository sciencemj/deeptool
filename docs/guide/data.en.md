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

## Hugging Face datasets

`datasets` is not a dependency of `deeptool` either. Install it when you need it.

```bash
uv run --with datasets jupyter lab
```

The batch convention absorbs both images and text. The only thing you write is
the `collate`.

```
image  batch = (image, label)                     → self(*batch[:-1]) = self(image)
text   batch = (input_ids, attention_mask, label) → self(*batch[:-1]) = self(ids, mask)
```

### Two things HF does differently

**It yields dicts.** A HF dataset gives you `{"image": …, "label": …}`.
`deeptool` reads `batch[:-1]` as inputs and `batch[-1]` as targets, so that
does not work as is.

```
KeyError: slice(None, -1, None)
```

**Images arrive as uint8 0-255.** `with_format("torch")` converts to tensors but
does not normalize. The `/255` and float conversion that torchvision's
`ToTensor()` performs is missing.

```
RuntimeError: mat1 and mat2 must have the same dtype, but got Byte and Float
```

The name makes this easy to miss. The shape is already `(N, 1, 28, 28)`,
channel-first, so that part needs nothing.

### Images

```python
import torch
from torch.utils.data import DataLoader, default_collate
from datasets import load_dataset


class HFMnist(dt.DataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.save_hyperparameters()
        d = load_dataset("ylecun/mnist")
        self.splits = {True: d["train"].with_format("torch"),
                       False: d["test"].with_format("torch")}

    def get_dataloader(self, train):
        def collate(examples):
            b = default_collate(examples)
            return b["image"].float() / 255, b["label"]
        return DataLoader(self.splits[train], self.batch_size,
                          shuffle=train, collate_fn=collate)
```

That one `collate` handles both problems — dict to tuple, and `/255` to
normalize. Everything after it is business as usual.

```python
data = HFMnist()
trainer = dt.Trainer(max_epochs=5, patience=3)
trainer.fit(model, data)          # model is a dt.Module subclass
trainer.restore_best()
trainer.predict(data).accuracy
```

### Text

You need a tokenizer. Also not a dependency of `deeptool`.

```bash
uv run --with datasets --with transformers jupyter lab
```

```python
from transformers import AutoTokenizer

TOK = AutoTokenizer.from_pretrained("distilbert-base-uncased")


class IMDB(dt.DataModule):
    def __init__(self, batch_size=16, max_length=128):
        super().__init__()
        self.save_hyperparameters()
        d = load_dataset("stanfordnlp/imdb")
        self.splits = {True: d["train"].shuffle(seed=0),
                       False: d["test"].shuffle(seed=0)}

    def get_dataloader(self, train):
        def collate(examples):
            enc = TOK([e["text"] for e in examples], truncation=True,
                      padding="max_length", max_length=self.max_length,
                      return_tensors="pt")
            labels = torch.tensor([e["label"] for e in examples])
            return enc["input_ids"], enc["attention_mask"], labels
        return DataLoader(self.splits[train], self.batch_size,
                          shuffle=train, collate_fn=collate)
```

No `with_format("torch")` here. Strings will not become tensors anyway, and the
tokenizer wants the raw `str`.

The batch now has three elements, and `deeptool` takes it unchanged.
`batch[:-1]` is `(input_ids, attention_mask)`, so `forward` just needs to accept
two arguments.

```python
class TextNet(dt.Module):
    def __init__(self, vocab_size=TOK.vocab_size, dim=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=TOK.pad_token_id)
        self.head = nn.Linear(dim, 2)

    def forward(self, input_ids, attention_mask):
        h = self.emb(input_ids) * attention_mask.unsqueeze(-1)
        return self.head(h.sum(1) / attention_mask.sum(1, keepdim=True))
```

### The format is not uniform

Do not read this far and conclude that all HF datasets look like this. **The
dict wrapper is the only thing they share.**

| Dataset | After `with_format("torch")` | After `default_collate` |
|---|---|---|
| `ylecun/mnist` | `Tensor`, `Tensor` | `(4,1,28,28) uint8`, `(4,) int64` |
| `stanfordnlp/imdb` | **`str`**, `Tensor` | **`list`**, `(4,) int64` |
| `rajpurkar/squad` | `str`×4, nested `dict` | `list`×4, nested `dict` |
| `Helsinki-NLP/opus_books` | `str`, nested `dict` | `list`, nested `dict` |

`with_format("torch")` **cannot turn strings into tensors.** It leaves them as
`str`. That is not a shortcoming of HF — the tokenizer differs per model, so it
cannot be decided at the dataset level.

QA and translation carry nested dicts that `default_collate` cannot batch into
tensors either. For those tasks the preprocessing pipeline is the real work and
the dataloader is the last step. It is not something `deeptool` can do for you.

!!! note "Why there is no `HFDataModule`"
    Installing `datasets` pulls in 35 packages; `transformers` pulls in 46.
    Keeping runtime dependencies at `torch`, `matplotlib` and `ipython` is a
    policy of this library, and the code it would save is six lines for images.

    The `collate` differs completely per modality, so there is nothing to
    unify anyway.

Both examples above were run on 2026-08-03. MNIST with 4,000/1,000 samples and a
linear model reached 0.857 accuracy in 5 epochs; IMDB with 800/400 samples and
embedding plus mean-pooling reached 0.6575 in 6 epochs. In both cases
`predict`, `restore_best` and `patience` worked without modification.

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
