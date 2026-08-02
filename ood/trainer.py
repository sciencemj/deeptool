"""학습 루프 — 디바이스 배치, 에폭 반복, 손실 집계, 조기 종료."""

import torch

from ood.board import ProgressBoard
from ood.checkpoint import BestSnapshot
from ood.core import HyperParameters
# 아래 셋은 Trainer 의 동명 메서드와 겹치므로 별칭으로 가져온다.
from ood.checkpoint import load_checkpoint as _load_checkpoint
from ood.checkpoint import save_checkpoint as _save_checkpoint
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

    def __init__(self, max_epochs, device=None, gradient_clip_val=0, plot=True,
                 snapshot_best=True, best_path=None, best_with_optim=False,
                 patience=None):
        self.save_hyperparameters()
        # patience=0 이면 최저점 epoch 에서도 epoch - best_epoch >= 0 이 참이 되어
        # 첫 epoch 직후 멈춘다. 의미가 없으므로 막는다.
        if patience is not None and patience < 1:
            raise ValueError(f"patience 는 1 이상이어야 합니다 (받은 값: {patience})")
        self.device = torch.device(device) if device is not None else default_device()
        self.board = ProgressBoard(xlabel="epoch", ylabel="loss") if plot else None
        self.history = {"train_loss": [], "val_loss": []}
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self._best = BestSnapshot(snapshot_best, best_path, best_with_optim)

    @property
    def best_val_loss(self):
        """최저 검증 손실. 아직 기록이 없으면 ``None``."""
        return self._best.val_loss

    @property
    def best_epoch(self):
        """최저 검증 손실을 낸 epoch. 아직 기록이 없으면 ``None``."""
        return self._best.epoch

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
        # 검증 데이터가 없으면 best_epoch 가 계속 None 이라 조기 종료가 영원히
        # 발동하지 않는다. 조용히 무시하면 왜 안 멈추는지 알 수 없으므로 막는다.
        if self.patience is not None and self.num_val_batches == 0:
            raise ValueError("patience 를 쓰려면 검증 데이터가 필요합니다.")
        self.prepare_model(model)
        self.materialize_lazy_parameters()
        self.optim = self.model.configure_optimizers()
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
            if self._should_stop_early():
                break
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
        val_loss = sum(losses) / len(losses)
        self.history["val_loss"].append(val_loss)
        self._best.update(val_loss, self.epoch, self.model, self.optim)

    def clip_gradients(self, grad_clip_val):
        params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, grad_clip_val)

    def _should_stop_early(self):
        """개선 없이 ``patience`` epoch 가 지났으면 True."""
        if self.patience is None or self.best_epoch is None:
            return False
        return self.epoch - self.best_epoch >= self.patience

    def restore_best(self):
        """최저 검증 손실 시점의 가중치로 되돌리고 그 epoch 를 반환한다."""
        if not hasattr(self, "model"):
            raise RuntimeError("아직 학습하지 않았습니다.")
        return self._best.restore(self.model)

    def save_checkpoint(self, path):
        """모델·optimizer 상태와 에폭·하이퍼파라미터를 한 파일로 저장한다."""
        _save_checkpoint(self.model, self.optim, self.epoch, path)

    @staticmethod
    def load_checkpoint(path, model, optim=None):
        """체크포인트를 ``model`` 에 in-place 로 복원한다.

        ``optim`` 을 주면 optimizer 상태까지 복원한다 (학습 재개용).
        주지 않으면 가중치만 복원한다 (추론용).
        저장된 ``epoch`` 과 ``hparams`` 를 담은 dict 를 반환한다.
        """
        return _load_checkpoint(path, model, optim)

    def predict(self, data, train=False, keep_inputs=False):
        """학습이 끝난 모델을 ``data`` 전체에 돌려 ``Predictions`` 를 반환한다.

        기본은 검증셋이고, ``train=True`` 면 학습셋을 쓴다.
        """
        loader = data.train_dataloader() if train else data.val_dataloader()
        return _predict(self.model, loader, self.device, keep_inputs)
