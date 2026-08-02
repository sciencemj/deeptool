import pytest
import torch

from deeptool.data import DataModule


class ToyData(DataModule):
    def __init__(self, n=20, batch_size=4):
        super().__init__()
        self.save_hyperparameters()
        self.X = torch.arange(n, dtype=torch.float32).reshape(n, 1)
        self.y = self.X * 2

    def get_dataloader(self, train):
        idx = slice(0, 16) if train else slice(16, None)
        return self.get_tensorloader((self.X, self.y), train, idx)


def test_get_dataloader_is_abstract():
    with pytest.raises(NotImplementedError):
        DataModule().get_dataloader(train=True)


def test_train_dataloader_requests_train_split(monkeypatch):
    seen = []
    data = DataModule()
    monkeypatch.setattr(data, "get_dataloader", lambda train: seen.append(train))

    data.train_dataloader()
    data.val_dataloader()

    assert seen == [True, False]


def test_get_tensorloader_yields_batches_of_the_right_shape():
    data = ToyData()
    batch = next(iter(data.train_dataloader()))

    assert len(batch) == 2
    assert batch[0].shape == (4, 1)
    assert batch[1].shape == (4, 1)


def test_get_tensorloader_honours_indices():
    data = ToyData()

    assert len(data.train_dataloader().dataset) == 16
    assert len(data.val_dataloader().dataset) == 4


def test_val_loader_is_not_shuffled():
    data = ToyData()
    first = torch.cat([x for x, _ in data.val_dataloader()])
    second = torch.cat([x for x, _ in data.val_dataloader()])

    assert torch.equal(first, second)


def test_subclass_batch_size_wins_over_default():
    """super().__init__() 을 먼저 부르고 save_hyperparameters 를 나중에 부르는 규약."""
    assert ToyData(batch_size=8).batch_size == 8
    assert DataModule().batch_size == 32
