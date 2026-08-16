# deeptool

[![CI](https://github.com/sciencemj/deeptool/actions/workflows/ci.yml/badge.svg)](https://github.com/sciencemj/deeptool/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/deeptool)](https://pypi.org/project/deeptool/)

📖 **[문서](https://sciencemj.github.io/deeptool/)** · [English](https://sciencemj.github.io/deeptool/en/)

주피터 노트북에서 PyTorch 모델을 객체지향으로 다루기 위한 얇은 보조 라이브러리.

모델은 유저가 PyTorch로 직접 작성한다. 이 라이브러리는 그 주변만 담당한다 —
하이퍼파라미터 자동 저장, 셀 간 메서드 추가, 학습 중 손실 곡선 라이브 렌더링,
디바이스 자동 선택, 학습 기록 영속화, 체크포인트.

## 설치

```bash
uv add deeptool     # uv 프로젝트에
pip install deeptool
```

```python
import deeptool as dt
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

trainer.save_checkpoint("linreg.pt")
```

전체 예제는 [`examples/quickstart.ipynb`](examples/quickstart.ipynb) 참고.

### 학습 기록과 실행 비교

`log_dir`을 주면 메타데이터와 완료된 에폭을 즉시 디스크에 남긴다.

```python
trainer = dt.Trainer(max_epochs=50, plot=False, log_dir="runs/exp1")
trainer.fit(model, data)
```

모델 안의 사용자 지표는 이름을 그대로 쓴다. 같은 에폭에서 여러 번 부르면
평균 한 점이 된다.

```python
self.log("iou", value)
```

실행마다 `meta.json`과 append-only `history.jsonl`이 생긴다. 스크립트에서는
`log_dir`을 준 경우에만 에폭당 한 줄도 출력한다. 기록 중이거나 중간에 멈춘
실행도 완료된 줄까지 읽고 비교할 수 있다.

```python
runs = dt.load_runs("runs")
figures = dt.plot_runs(runs)
```

`plot_runs`는 지표마다 Figure 하나를 만들고 그 지표가 있는 실행만 겹쳐 그린다.
모델마다 지표 이름이 달라도 별도 스키마가 필요 없다.

LLM처럼 에폭보다 optimizer update 횟수가 중요한 경우에는 `max_steps`를 쓴다.
`max_epochs`와 `max_steps` 중 정확히 하나만 지정해야 한다.

```python
trainer = dt.Trainer(
    max_steps=10_000,
    log_every_n_steps=50,
    val_every_n_steps=500,
    scheduler_interval="step",
    patience=4,
    monitor="val_loss",
    log_dir="runs/llm-1",
)
```

step 모드의 기록은 `epoch` 대신 1부터 시작하는 `step`을 쓴다. 유한하고 비어
있지 않은 train DataLoader를 끝까지 돌면 자동으로 다시 순회한다. 검증은
`val_every_n_steps` 간격과 마지막 step에 실행하며, 생략하면 마지막에만 실행한다.
`best_step`과 `restore_best()`도 같은 optimizer step을 가리킨다.

### 학습률 스케줄러

scheduler는 optimizer와 함께 반환한다.

```python
def configure_optimizers(self):
    optim = torch.optim.Adam(self.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
    return optim, scheduler
```

`scheduler_interval="auto"`는 에폭 학습에서 에폭마다, step 학습에서 optimizer
update마다 일반 scheduler의 `step()`을 부른다. 필요하면 `"epoch"` 또는
`"step"`으로 고정할 수 있다. `ReduceLROnPlateau`는 이 간격과 무관하게 검증 뒤
`step(val_loss)`로 호출된다. AMP는 아직 지원하지 않는다.

### 조기 종료와 최적 가중치

개선이 멈출 때까지 돌리고 가장 좋았던 가중치를 쓴다.

```python
trainer = dt.Trainer(max_epochs=100, patience=5)
trainer.fit(model, data)

len(trainer.history["val_loss"])             # 24 — 100까지 안 감
trainer.best_epoch, trainer.best_val_loss    # (18, 0.2913)

trainer.restore_best()                       # 18 을 반환
```

IoU·accuracy처럼 클수록 좋은 지표는 검증 단계에서 `self.log("iou", value)`로
기록하고 `Trainer(monitor="iou", mode="max")`로 지정한다. 같은 기준이 best
snapshot, `restore_best()`, 조기 종료를 모두 제어한다. 기본은
`monitor="val_loss", mode="min"`이다.

`fit()` 은 가중치를 자동으로 되돌리지 않는다. `restore_best()` 를 부르기 전까지는
마지막 epoch 상태이므로 두 시점의 성능을 비교할 수 있다.

기본은 메모리 스냅샷이다. 파일로 남기려면:

```python
dt.Trainer(max_epochs=100, patience=5, best_path="best.pt")
```

파일에는 모델 가중치만 들어간다. optimizer 상태는 `restore_best()` 가 읽지 않는데
Adam 기준 모델의 2배라 매 epoch 쓰면 낭비다. 최저점부터 학습을 재개할 계획이면
`best_with_optim=True` 로 전체 체크포인트를 남긴다.

| 인자 | 기본 | 의미 |
|---|---|---|
| `snapshot_best` | `True` | 스냅샷을 만들 것인가 |
| `best_path` | `None` | `None` 이면 메모리, 경로면 파일 |
| `best_with_optim` | `False` | 파일에 optimizer 상태도 넣을 것인가 |
| `patience` | `None` | 몇 번의 monitor 확인 동안 개선이 없으면 멈출 것인가 |
| `monitor` | `"val_loss"` | best/조기 종료에 사용할 지표 이름 |
| `mode` | `"min"` | `min` 또는 `max` |

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
| `dt.add_to_class(Class)` | 데코레이트한 함수를 `Class` 의 메서드로 등록 |
| `dt.HyperParameters` | `save_hyperparameters()` 로 `__init__` 인자를 속성 + `hparams` 로 저장 |
| `dt.DataModule` | `get_dataloader(train)` 하나만 구현하면 되는 데이터 규약 |
| `dt.Module` | `forward`/`loss`/`configure_optimizers` 를 채우는 모델 규약 |
| `dt.Trainer` | `fit(model, data)`, `predict(data)`, `restore_best()`, `save_checkpoint`, `load_checkpoint`, `history`, `best_score`, `best_epoch`, `best_step`, `best_val_loss` |
| `dt.predict` | 모델과 dataloader 를 받아 데이터셋 전체 예측을 모은다 |
| `dt.Predictions` | 예측 결과. `preds`·`probs`·`confidence`·`correct`·`accuracy` |
| `dt.ProgressBoard` | 라이브 손실 곡선. `Trainer(plot=True)` 가 자동으로 만든다 |
| `dt.RunRecorder` | 한 실행의 `meta.json`과 `history.jsonl` 기록 |
| `dt.load_runs` | 여러 실행의 자유형 JSONL 지표를 로드 |
| `dt.plot_runs` | 지표별 실행 비교 Figure 목록 생성 |
| `dt.default_device()` | `cuda` → `mps` → `cpu` |

## 개발

```bash
uv run pytest
```

## 라이센스

MIT. `LICENSE` 참고.

설계는 [d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en)의 `d2l/torch.py`를 참고했다.
해당 샘플 코드는 modified MIT(`LICENSE-SAMPLECODE`)로 배포된다.
