import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import ood as od


class SyntheticRegression(od.DataModule):
    def __init__(self, n=200, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(0)
        self.X = torch.randn(n, 2)
        self.y = self.X @ torch.tensor([[2.0], [-3.4]]) + 4.2

    def get_dataloader(self, train):
        idx = slice(0, 160) if train else slice(160, None)
        return self.get_tensorloader((self.X, self.y), train, idx)


class LinearRegression(od.Module):
    def __init__(self, lr=0.03):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)


@od.add_to_class(LinearRegression)
def loss(self, y_hat, y):
    return F.mse_loss(y_hat, y)


@od.add_to_class(LinearRegression)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)


def test_quickstart_trains_end_to_end():
    trainer = od.Trainer(max_epochs=20, plot=False)
    history = trainer.fit(LinearRegression(), SyntheticRegression())

    assert history["train_loss"][-1] < 0.5


def test_quickstart_notebook_is_valid_json():
    path = Path(__file__).parent.parent / "examples" / "quickstart.ipynb"
    nb = json.loads(path.read_text())

    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 4
