"""학습이 끝난 모델의 사후 평가 — 데이터셋 전체 예측을 한 번에 모은다."""

import torch


class Predictions:
    """``predict`` 의 결과.

    원본 ``outputs`` 와 ``targets`` 만 저장하고 나머지는 속성으로 파생한다.
    속성은 캐시하지 않는다 — 무효화 규칙이 생기는 것보다 매번 계산하는 편이 싸다.

    ``preds``·``probs``·``confidence``·``correct``·``accuracy`` 는 **분류 전용**이다.
    회귀 모델이면 ``outputs`` 를 직접 쓴다.
    """

    def __init__(self, outputs, targets, inputs=None):
        self.outputs = outputs
        self.targets = targets
        self.inputs = inputs

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        body = f"n={len(self)} outputs={tuple(self.outputs.shape)}"
        # 회귀 결과에 accuracy 를 계산하면 브로드캐스트로 (N, N) 텐서가 만들어진다.
        # shape 이 맞을 때만 붙인다.
        if self.preds.shape == self.targets.shape:
            body += f" accuracy={self.accuracy:.4f}"
        return f"<Predictions {body}>"

    @property
    def preds(self):
        return self.outputs.argmax(dim=-1)

    @property
    def probs(self):
        return self.outputs.softmax(dim=-1)

    @property
    def confidence(self):
        return self.probs.max(dim=-1).values

    @property
    def correct(self):
        preds = self.preds
        if preds.shape != self.targets.shape:
            raise ValueError(
                f"preds shape {tuple(preds.shape)} 와 targets shape "
                f"{tuple(self.targets.shape)} 가 다릅니다. "
                "분류 전용 속성입니다 — 회귀 모델이면 outputs 를 직접 쓰세요."
            )
        return preds == self.targets

    @property
    def accuracy(self):
        return self.correct.float().mean().item()


@torch.no_grad()
def predict(model, dataloader, device=None, keep_inputs=False):
    """``dataloader`` 전체를 추론해 ``Predictions`` 로 모은다.

    배치 규약은 ``Module.training_step`` 과 같다: ``batch[:-1]`` 이 입력,
    ``batch[-1]`` 이 정답이다. ``inputs`` 로 보관하는 것은 ``batch[0]`` 이다.

    ``device`` 가 ``None`` 이면 모델 파라미터가 올라가 있는 디바이스를 쓴다.

    결과는 즉시 CPU 로 회수한다. 그러지 않으면 데이터셋 전체가 가속기 메모리에
    쌓이고, matplotlib 같은 downstream 코드도 CPU 텐서를 기대한다.

    ``keep_inputs`` 기본값은 ``False`` 다. 입력 텐서는 출력보다 훨씬 크다
    (28×28 이미지 1만장 = 31MB vs 출력 400KB).

    ``model.eval()`` 을 호출하고 이전 모드를 복원하지 않는다. ``Trainer.fit_epoch``
    이 매 에폭 ``model.train()`` 을 다시 부르므로 학습 재개에 영향이 없다.
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
