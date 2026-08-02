"""학습 루프 — 디바이스 배치, 에폭 반복, 손실 집계."""

import torch

from ood.board import ProgressBoard
from ood.core import HyperParameters
# 메서드 이름과 겹치므로 별칭으로 가져온다.
from ood.evaluate import predict as _predict


def default_device():
    """사용 가능한 가속기를 cuda → mps → cpu 순으로 고른다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer(HyperParameters):
    """``Module`` 과 ``DataModule`` 을 받아 학습을 돌린다."""

    def __init__(self, max_epochs, device=None, gradient_clip_val=0, plot=True):
        self.save_hyperparameters()
        self.device = torch.device(device) if device is not None else default_device()
        self.board = ProgressBoard(xlabel="epoch", ylabel="loss") if plot else None
        self.history = {"train_loss": [], "val_loss": []}
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (
            len(self.val_dataloader) if self.val_dataloader is not None else 0
        )

    def prepare_model(self, model):
        model.trainer = self
        model.board = self.board
        self.model = model.to(self.device)

    def prepare_batch(self, batch):
        return [a.to(self.device) for a in batch]

    def materialize_lazy_parameters(self):
        """LazyLinear 등을 optimizer 생성 전에 더미 forward 로 실체화한다."""
        batch = self.prepare_batch(next(iter(self.train_dataloader)))
        with torch.no_grad():
            self.model(*batch[:-1])

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.materialize_lazy_parameters()
        self.optim = self.model.configure_optimizers()
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
        return self.history

    def fit_epoch(self):
        self.model.train()
        losses = []
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            loss.backward()
            if self.gradient_clip_val > 0:
                self.clip_gradients(self.gradient_clip_val)
            self.optim.step()
            self.train_batch_idx += 1
            losses.append(loss.detach().cpu().item())
        self.history["train_loss"].append(sum(losses) / len(losses))

        if self.num_val_batches == 0:
            return
        self.model.eval()
        losses = []
        for batch in self.val_dataloader:
            with torch.no_grad():
                loss = self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1
            losses.append(loss.detach().cpu().item())
        self.history["val_loss"].append(sum(losses) / len(losses))

    def clip_gradients(self, grad_clip_val):
        params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, grad_clip_val)

    def save_checkpoint(self, path):
        """모델·optimizer 상태와 에폭·하이퍼파라미터를 한 파일로 저장한다."""
        torch.save(
            {
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
                "epoch": self.epoch,
                "hparams": getattr(self.model, "hparams", {}),
            },
            path,
        )

    @staticmethod
    def load_checkpoint(path, model, optim=None):
        """체크포인트를 ``model`` 에 in-place 로 복원한다.

        ``optim`` 을 주면 optimizer 상태까지 복원한다 (학습 재개용).
        주지 않으면 가중치만 복원한다 (추론용).
        저장된 ``epoch`` 과 ``hparams`` 를 담은 dict 를 반환한다.

        ``hparams`` 는 임의의 파이썬 객체일 수 있어 ``weights_only=False`` 로
        읽는다. 신뢰할 수 있는 체크포인트만 로드하라.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if optim is not None:
            optim.load_state_dict(ckpt["optim"])
        return {"epoch": ckpt["epoch"], "hparams": ckpt["hparams"]}

    def predict(self, data, train=False, keep_inputs=False):
        """학습이 끝난 모델을 ``data`` 전체에 돌려 ``Predictions`` 를 반환한다.

        기본은 검증셋이고, ``train=True`` 면 학습셋을 쓴다.
        """
        loader = data.train_dataloader() if train else data.val_dataloader()
        return _predict(self.model, loader, self.device, keep_inputs)
