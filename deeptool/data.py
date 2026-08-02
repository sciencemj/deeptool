"""Data loading contract: training and validation loaders on one object."""

from collections.abc import Sequence

import torch
from torch.utils import data as torch_data

from deeptool.core import HyperParameters


class DataModule(HyperParameters):
    """Base class supplying training and validation dataloaders.

    Subclasses implement `get_dataloader(train)` and nothing else.

    Call `super().__init__()` before `save_hyperparameters()`, or the parent's
    defaults overwrite the subclass arguments:

    ```python
    class ToyData(DataModule):
        def __init__(self, batch_size=32):
            super().__init__()
            self.save_hyperparameters()
    ```
    """

    def __init__(self, root: str = "../data", num_workers: int = 0,
                 batch_size: int = 32) -> None:
        self.save_hyperparameters()

    def get_dataloader(self, train: bool) -> torch_data.DataLoader:
        raise NotImplementedError

    def train_dataloader(self) -> torch_data.DataLoader:
        return self.get_dataloader(train=True)

    def val_dataloader(self) -> torch_data.DataLoader:
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors: Sequence[torch.Tensor], train: bool,
                         indices: slice = slice(0, None)) -> torch_data.DataLoader:
        """Wrap a tuple of tensors in a `DataLoader`.

        Args:
            tensors: Tensors sliced together; the last one holds the targets.
            train: Shuffles when true.
            indices: Slice applied to every tensor, for splitting training from
                validation.

        Returns:
            A `torch.utils.data.DataLoader` over a `TensorDataset`.
        """
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch_data.TensorDataset(*tensors)
        return torch_data.DataLoader(
            dataset, self.batch_size, shuffle=train,
            num_workers=self.num_workers,
        )
