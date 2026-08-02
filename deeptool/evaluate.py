"""Post-hoc evaluation: collect predictions over a whole dataset at once."""

from collections.abc import Iterable, Sequence

import torch


class Predictions:
    """The result of `predict`.

    Stores only the raw `outputs` and `targets`; everything else is derived as a
    property and recomputed each time rather than cached.

    `preds`, `probs`, `confidence`, `correct` and `accuracy` are
    **classification only**. For a regression model, use `outputs` directly.

    Attributes:
        outputs: Raw model output, shape `(N, ...)`. Always on CPU.
        targets: Ground truth, shape `(N, ...)`. Always on CPU.
        inputs: Input tensors, present only when `predict` was called with
            `keep_inputs=True`. Otherwise `None`.
    """

    def __init__(self, outputs: torch.Tensor, targets: torch.Tensor,
                 inputs: torch.Tensor | None = None) -> None:
        self.outputs = outputs
        self.targets = targets
        self.inputs = inputs

    def __len__(self) -> int:
        return len(self.targets)

    def __repr__(self) -> str:
        body = f"n={len(self)} outputs={tuple(self.outputs.shape)}"
        # 회귀 결과에 accuracy 를 계산하면 브로드캐스트로 (N, N) 텐서가 만들어진다.
        # shape 이 맞을 때만 붙인다.
        if self.preds.shape == self.targets.shape:
            body += f" accuracy={self.accuracy:.4f}"
        return f"<Predictions {body}>"

    @property
    def preds(self) -> torch.Tensor:
        """Predicted class per sample, `outputs.argmax(dim=-1)`."""
        return self.outputs.argmax(dim=-1)

    @property
    def probs(self) -> torch.Tensor:
        """Class probabilities, `outputs.softmax(dim=-1)`."""
        return self.outputs.softmax(dim=-1)

    @property
    def confidence(self) -> torch.Tensor:
        """Probability assigned to the predicted class."""
        return self.probs.max(dim=-1).values

    @property
    def correct(self) -> torch.Tensor:
        """Boolean tensor of whether each prediction matches its target.

        Raises:
            ValueError: If `preds` and `targets` have different shapes, which
                means this is not a classification result.
        """
        preds = self.preds
        if preds.shape != self.targets.shape:
            raise ValueError(
                f"preds shape {tuple(preds.shape)} does not match targets shape "
                f"{tuple(self.targets.shape)}. This is a classification-only "
                "property; use outputs directly for regression."
            )
        return preds == self.targets

    @property
    def accuracy(self) -> float:
        """Fraction of correct predictions, as a plain float."""
        return self.correct.float().mean().item()


@torch.no_grad()
def predict(model: torch.nn.Module,
            dataloader: Iterable[Sequence[torch.Tensor]],
            device: torch.device | str | None = None,
            keep_inputs: bool = False) -> Predictions:
    """Run the model over an entire dataloader and collect per-sample results.

    Results are moved to CPU as they arrive, so the dataset never accumulates in
    accelerator memory and downstream code gets the CPU tensors it expects.

    Puts the model in eval mode and leaves it there. `Trainer.fit_epoch` calls
    `model.train()` at the start of every epoch, so resuming training is safe.

    Args:
        model: A model following the `Module` batch convention — `batch[:-1]`
            are inputs and `batch[-1]` are targets.
        dataloader: Batches to run through the model.
        device: Where to run inference. Defaults to the model's own device.
        keep_inputs: Also collect `batch[0]`. Off by default because inputs are
            much larger than outputs — 10,000 28x28 images is 31MB against 400KB
            of logits.

    Returns:
        A `Predictions` holding CPU tensors.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    outputs, targets, inputs = [], [], []
    for batch in dataloader:
        batch = [a.to(device) for a in batch]
        outputs.append(model(*batch[:-1]).cpu())
        targets.append(batch[-1].cpu())
        if keep_inputs:
            inputs.append(batch[0].cpu())

    return Predictions(
        torch.cat(outputs),
        torch.cat(targets),
        torch.cat(inputs) if keep_inputs else None,
    )
