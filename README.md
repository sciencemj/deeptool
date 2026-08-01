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

## API

| 이름 | 역할 |
|---|---|
| `od.add_to_class(Class)` | 데코레이트한 함수를 `Class` 의 메서드로 등록 |
| `od.HyperParameters` | `save_hyperparameters()` 로 `__init__` 인자를 속성 + `hparams` 로 저장 |
| `od.DataModule` | `get_dataloader(train)` 하나만 구현하면 되는 데이터 규약 |
| `od.Module` | `forward`/`loss`/`configure_optimizers` 를 채우는 모델 규약 |
| `od.Trainer` | `fit(model, data)`, `save_checkpoint`, `load_checkpoint`, `history` |
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
