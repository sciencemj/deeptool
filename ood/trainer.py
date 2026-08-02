"""학습 루프 — 디바이스 배치, 에폭 반복, 손실 집계."""

import copy
import os

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


def _atomic_save(payload, path):
    """임시 파일에 쓴 뒤 원자적으로 교체한다.

    최적 스냅샷은 개선될 때마다 같은 경로를 덮어쓴다. 쓰는 도중 중단되면
    그때까지 쌓은 최적 가중치를 통째로 잃으므로, 교체가 원자적이어야 한다.
    ``os.replace`` 는 POSIX 와 Windows 양쪽에서 원자적이며, 실패하면
    이전 파일이 그대로 남는다.
    """
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


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
        self.best_val_loss = None
        self.best_epoch = None
        self._best_state = None

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
        self._track_best(val_loss)

    def clip_gradients(self, grad_clip_val):
        params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, grad_clip_val)

    def _should_stop_early(self):
        """개선 없이 ``patience`` epoch 가 지났으면 True."""
        if self.patience is None or self.best_epoch is None:
            return False
        return self.epoch - self.best_epoch >= self.patience

    def _track_best(self, val_loss):
        """검증 손실이 최저를 갱신하면 기록하고 스냅샷을 뜬다.

        ``snapshot_best=False`` 여도 ``best_val_loss``·``best_epoch`` 는 계속
        기록한다. float 비교라 비용이 없고, "몇 번째가 최저였는가" 는 그 자체로
        쓸모가 있다.
        """
        if self.best_val_loss is not None and val_loss >= self.best_val_loss:
            return
        self.best_val_loss = val_loss
        self.best_epoch = self.epoch
        if not self.snapshot_best:
            return
        if self.best_path is None:
            self._best_state = copy.deepcopy(self.model.state_dict())
            return
        # optimizer 상태는 restore_best() 가 읽지 않는다. Adam 기준 모델의 2배라
        # 매 개선마다 쓰면 낭비이므로 기본값은 가중치 전용이다.
        if self.best_with_optim:
            payload = self._checkpoint_payload()
        else:
            payload = {
                "model": self.model.state_dict(),
                "epoch": self.epoch,
                "val_loss": val_loss,
            }
        _atomic_save(payload, self.best_path)

    def restore_best(self):
        """최저 검증 손실 시점의 가중치로 되돌리고 그 epoch 를 반환한다.

        모델 가중치만 되돌린다. optimizer 상태는 건드리지 않는다 — 목적이
        "가장 좋은 모델로 평가·추론" 이지 학습 재개가 아니다.
        """
        if not hasattr(self, "model"):
            raise RuntimeError("아직 학습하지 않았습니다.")
        if self.best_epoch is None:
            raise RuntimeError("검증 데이터가 없어 스냅샷이 없습니다.")
        if not self.snapshot_best:
            raise RuntimeError(
                f"snapshot_best=False 로 학습해 스냅샷이 없습니다. "
                f"(최저점은 epoch {self.best_epoch}, "
                f"val_loss {self.best_val_loss:.4f} 이었습니다)"
            )
        if self.best_path is None:
            self.model.load_state_dict(self._best_state)
        else:
            ckpt = torch.load(self.best_path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["model"])
        return self.best_epoch

    def _checkpoint_payload(self):
        """학습 재개가 가능한 전체 체크포인트 페이로드."""
        return {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "epoch": self.epoch,
            "hparams": getattr(self.model, "hparams", {}),
        }

    def save_checkpoint(self, path):
        """모델·optimizer 상태와 에폭·하이퍼파라미터를 한 파일로 저장한다."""
        torch.save(self._checkpoint_payload(), path)

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
