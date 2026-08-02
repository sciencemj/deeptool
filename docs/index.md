# deeptool

주피터 노트북에서 PyTorch 모델을 객체지향으로 다루기 위한 얇은 보조 라이브러리.

모델은 당신이 PyTorch로 직접 쓴다. `deeptool`은 그 주변만 담당한다.

```bash
pip install deeptool
```

## 30초 예제

```python
import torch
from torch import nn
from torch.nn import functional as F

import deeptool as dt


class SyntheticRegression(dt.DataModule):
    def __init__(self, n=200, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(0)
        self.X = torch.randn(n, 2)
        self.y = self.X @ torch.tensor([[2.0], [-3.4]]) + 4.2

    def get_dataloader(self, train):
        idx = slice(0, 160) if train else slice(160, None)
        return self.get_tensorloader((self.X, self.y), train, idx)


class LinearRegression(dt.Module):
    def __init__(self, lr=0.03):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
```

다음 셀에서 메서드를 덧붙인다. 클래스를 다시 정의할 필요가 없다.

```python
@dt.add_to_class(LinearRegression)
def loss(self, y_hat, y):
    return F.mse_loss(y_hat, y)


@dt.add_to_class(LinearRegression)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

학습을 돌리면 손실 곡선이 셀 출력에 실시간으로 갱신된다.

```python
trainer = dt.Trainer(max_epochs=20)
trainer.fit(LinearRegression(), SyntheticRegression())
```

## 담당하는 것

| | |
|---|---|
| 하이퍼파라미터 저장 | `save_hyperparameters()` 가 `__init__` 인자를 속성과 `hparams` 로 |
| 셀 간 메서드 추가 | `@add_to_class` 로 클래스 재정의 없이 |
| 라이브 손실 곡선 | 학습 중 셀 안에서 제자리 갱신 |
| 디바이스 자동 선택 | `cuda` → `mps` → `cpu` |
| 최적 가중치 · 조기 종료 | 검증 손실 최저점 스냅샷, `patience` |
| 사후 평가 | `trainer.predict(data)` 로 샘플별 예측 수집 |
| 체크포인트 | 저장 · 복원 |

## 담당하지 않는 것

모델 zoo, 데이터셋 번들, 분산 학습, 이미지 시각화, 로깅 백엔드 연동.

런타임 의존성은 `torch`, `matplotlib`, `ipython` 셋뿐이다. 이 목록은 늘리지 않는다.

## 어디서 시작할까

- [퀵스타트](quickstart.md) — 선형회귀 전체 흐름
- [모델](guide/module.md) — `Module` 과 `add_to_class`
- [데이터](guide/data.md) — `DataModule`
- [학습기](guide/trainer.md) — `Trainer`, 디바이스, `history`
- [최적 가중치와 조기 종료](guide/best.md) — `restore_best`, `patience`
- [사후 평가](guide/evaluate.md) — `predict`, `Predictions`
- [API](api.md) — 전체 레퍼런스

## 라이센스

MIT. 설계는 [d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en)의 `d2l/torch.py`를 참고했다.
