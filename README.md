# ood

[![CI](https://github.com/sciencemj/ood-dl/actions/workflows/ci.yml/badge.svg)](https://github.com/sciencemj/ood-dl/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ood-dl)](https://pypi.org/project/ood-dl/)

주피터 노트북에서 PyTorch 모델을 객체지향으로 다루기 위한 얇은 보조 라이브러리.

모델은 유저가 PyTorch로 직접 작성한다. 이 라이브러리는 그 주변만 담당한다 —
하이퍼파라미터 자동 저장, 셀 간 메서드 추가, 학습 중 손실 곡선 라이브 렌더링,
디바이스 자동 선택, 체크포인트.

## 설치

```bash
pip install ood-dl
```

배포 이름은 `ood-dl`, import 이름은 `ood` 다. PyPI 에 `ood` 이름이 이미 쓰이고 있어서다.

```python
import ood as od
```

이 저장소에서 직접 개발하려면:

```bash
uv sync
```

## 퀵스타트

```python
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
```

다음 셀에서 메서드를 덧붙인다. 클래스를 다시 정의할 필요가 없다.

```python
@od.add_to_class(LinearRegression)
def loss(self, y_hat, y):
    return F.mse_loss(y_hat, y)


@od.add_to_class(LinearRegression)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

학습을 돌리면 손실 곡선이 셀 출력에 실시간으로 갱신된다.

```python
trainer = od.Trainer(max_epochs=20)
trainer.fit(LinearRegression(), SyntheticRegression())

trainer.save_checkpoint("linreg.pt")
```

전체 예제는 [`examples/quickstart.ipynb`](examples/quickstart.ipynb) 참고.

### 조기 종료와 최적 가중치

개선이 멈출 때까지 돌리고 가장 좋았던 가중치를 쓴다.

```python
trainer = od.Trainer(max_epochs=100, patience=5)
trainer.fit(model, data)

len(trainer.history["val_loss"])             # 24 — 100까지 안 감
trainer.best_epoch, trainer.best_val_loss    # (18, 0.2913)

trainer.restore_best()                       # 18 을 반환
```

`fit()` 은 가중치를 자동으로 되돌리지 않는다. `restore_best()` 를 부르기 전까지는
마지막 epoch 상태이므로 두 시점의 성능을 비교할 수 있다.

기본은 메모리 스냅샷이다. 파일로 남기려면:

```python
od.Trainer(max_epochs=100, patience=5, best_path="best.pt")
```

파일에는 모델 가중치만 들어간다. optimizer 상태는 `restore_best()` 가 읽지 않는데
Adam 기준 모델의 2배라 매 epoch 쓰면 낭비다. 최저점부터 학습을 재개할 계획이면
`best_with_optim=True` 로 전체 체크포인트를 남긴다.

| 인자 | 기본 | 의미 |
|---|---|---|
| `snapshot_best` | `True` | 스냅샷을 만들 것인가 |
| `best_path` | `None` | `None` 이면 메모리, 경로면 파일 |
| `best_with_optim` | `False` | 파일에 optimizer 상태도 넣을 것인가 |
| `patience` | `None` | 몇 epoch 개선이 없으면 멈출 것인가 |

### 학습 후 평가

```python
p = trainer.predict(data)        # 검증셋 전체 추론

p.accuracy                       # 0.8837
p.preds                          # 샘플별 예측 클래스
p.confidence                     # 예측 확신도
p.correct                        # 맞췄는지 여부 (bool 텐서)

p = trainer.predict(data, keep_inputs=True)
p.inputs[~p.correct]             # 틀린 샘플의 입력 — 시각화에 쓴다
```

`preds`·`probs`·`confidence`·`correct`·`accuracy` 는 분류 전용이다.
회귀 모델이면 `p.outputs` 를 직접 쓴다.

## API

| 이름 | 역할 |
|---|---|
| `od.add_to_class(Class)` | 데코레이트한 함수를 `Class` 의 메서드로 등록 |
| `od.HyperParameters` | `save_hyperparameters()` 로 `__init__` 인자를 속성 + `hparams` 로 저장 |
| `od.DataModule` | `get_dataloader(train)` 하나만 구현하면 되는 데이터 규약 |
| `od.Module` | `forward`/`loss`/`configure_optimizers` 를 채우는 모델 규약 |
| `od.Trainer` | `fit(model, data)`, `predict(data)`, `restore_best()`, `save_checkpoint`, `load_checkpoint`, `history`, `best_epoch`, `best_val_loss` |
| `od.predict` | 모델과 dataloader 를 받아 데이터셋 전체 예측을 모은다 |
| `od.Predictions` | 예측 결과. `preds`·`probs`·`confidence`·`correct`·`accuracy` |
| `od.ProgressBoard` | 라이브 손실 곡선. `Trainer(plot=True)` 가 자동으로 만든다 |
| `od.default_device()` | `cuda` → `mps` → `cpu` |

## 개발

```bash
uv run pytest
```

## 라이센스

MIT. `LICENSE` 참고.

설계는 [d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en)의 `d2l/torch.py`를 참고했다.
해당 샘플 코드는 modified MIT(`LICENSE-SAMPLECODE`)로 배포된다.
