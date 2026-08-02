"""체크포인트 저장·복원과 최저 검증 손실 시점 스냅샷."""

import copy
import os

import torch


def atomic_save(payload, path):
    """임시 파일에 쓴 뒤 원자적으로 교체한다.

    최적 스냅샷은 개선될 때마다 같은 경로를 덮어쓴다. 쓰는 도중 중단되면
    그때까지 쌓은 최적 가중치를 통째로 잃으므로, 교체가 원자적이어야 한다.
    ``os.replace`` 는 POSIX 와 Windows 양쪽에서 원자적이며, 실패하면
    이전 파일이 그대로 남는다.
    """
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def checkpoint_payload(model, optim, epoch):
    """학습 재개가 가능한 전체 체크포인트 페이로드."""
    return {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "hparams": getattr(model, "hparams", {}),
    }


def save_checkpoint(model, optim, epoch, path):
    """모델·optimizer 상태와 에폭·하이퍼파라미터를 한 파일로 저장한다."""
    torch.save(checkpoint_payload(model, optim, epoch), path)


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


class BestSnapshot:
    """검증 손실이 최저였던 시점의 모델 가중치를 보관한다.

    ``path`` 가 ``None`` 이면 메모리에 ``deepcopy`` 로, 경로면 그 파일에 둔다.

    ``enabled=False`` 여도 ``val_loss``·``epoch`` 는 계속 기록한다. float 비교라
    비용이 없고, "몇 번째가 최저였는가" 는 그 자체로 쓸모가 있다. 끄는 것은
    복사·쓰기뿐이다.
    """

    def __init__(self, enabled=True, path=None, with_optim=False):
        self.enabled = enabled
        self.path = path
        self.with_optim = with_optim
        self.val_loss = None
        self.epoch = None
        self._state = None

    def update(self, val_loss, epoch, model, optim):
        """최저를 갱신했으면 기록하고 스냅샷을 남긴다."""
        if self.val_loss is not None and val_loss >= self.val_loss:
            return
        self.val_loss = val_loss
        self.epoch = epoch
        if not self.enabled:
            return
        if self.path is None:
            self._state = copy.deepcopy(model.state_dict())
            return
        # optimizer 상태는 restore() 가 읽지 않는다. Adam 기준 모델의 2배라
        # 매 개선마다 쓰면 낭비이므로 기본값은 가중치 전용이다.
        if self.with_optim:
            payload = checkpoint_payload(model, optim, epoch)
        else:
            payload = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }
        atomic_save(payload, self.path)

    def restore(self, model):
        """최저 시점 가중치로 ``model`` 을 되돌리고 그 epoch 를 반환한다.

        모델 가중치만 되돌린다. optimizer 상태는 건드리지 않는다 — 목적이
        "가장 좋은 모델로 평가·추론" 이지 학습 재개가 아니다.
        """
        if self.epoch is None:
            raise RuntimeError("검증 데이터가 없어 스냅샷이 없습니다.")
        if not self.enabled:
            raise RuntimeError(
                f"snapshot_best=False 로 학습해 스냅샷이 없습니다. "
                f"(최저점은 epoch {self.epoch}, "
                f"val_loss {self.val_loss:.4f} 이었습니다)"
            )
        if self.path is None:
            model.load_state_dict(self._state)
        else:
            ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
        return self.epoch
