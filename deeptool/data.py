"""데이터 로딩 규약 — 학습/검증 dataloader 를 한 객체에 묶는다."""

from torch.utils import data as torch_data

from deeptool.core import HyperParameters


class DataModule(HyperParameters):
    """학습·검증 dataloader 를 제공하는 베이스 클래스.

    서브클래스는 ``get_dataloader(train)`` 하나만 구현하면 된다::

        class ToyData(DataModule):
            def __init__(self, batch_size=32):
                super().__init__()           # 부모 기본값 먼저
                self.save_hyperparameters()  # 그 다음 내 인자로 덮어쓴다
                ...

    ``super().__init__()`` 을 먼저 부르고 ``save_hyperparameters()`` 를
    나중에 불러야 서브클래스 인자가 부모 기본값을 이긴다.
    """

    def __init__(self, root="../data", num_workers=0, batch_size=32):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        """텐서 묶음을 ``DataLoader`` 로 감싼다. ``train`` 이면 셔플한다."""
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch_data.TensorDataset(*tensors)
        return torch_data.DataLoader(
            dataset, self.batch_size, shuffle=train,
            num_workers=self.num_workers,
        )
