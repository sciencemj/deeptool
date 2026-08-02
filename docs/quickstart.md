# 퀵스타트

선형회귀 하나를 끝까지 돌리며 `deeptool`의 다섯 조각을 본다.
같은 내용을 노트북으로 실행하려면
[`examples/quickstart.ipynb`](https://github.com/sciencemj/deeptool/blob/main/examples/quickstart.ipynb)를 열면 된다.

```python
import torch
from torch import nn
from torch.nn import functional as F

import deeptool as dt
```

## 1. 데이터

`DataModule`을 상속하고 `get_dataloader(train)` 하나만 구현한다.
학습·검증 분리는 인덱스 슬라이스로 한다.

```python
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


data = SyntheticRegression()
data.hparams
```

```
{'n': 200, 'batch_size': 32}
```

`save_hyperparameters()`가 `__init__` 인자를 속성으로 만들고 `hparams`에도 담았다.
`self.batch_size`를 따로 대입하지 않았는데 `get_tensorloader`가 쓸 수 있는 이유다.

## 2. 모델

`Module`을 상속하고 `self.net`을 정의한다. `forward`는 자동으로 위임된다.

`nn.LazyLinear`는 입력 크기를 안 적어도 된다. 첫 forward에서 결정된다.

```python
class LinearRegression(dt.Module):
    def __init__(self, lr=0.03):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
```

## 3. 나중 셀에서 메서드 덧붙이기

여기가 노트북 작업의 핵심이다. `loss`와 `configure_optimizers`를 깜빡했어도
**클래스 정의 셀로 돌아가 다시 실행할 필요가 없다.**

```python
@dt.add_to_class(LinearRegression)
def loss(self, y_hat, y):
    return F.mse_loss(y_hat, y)


@dt.add_to_class(LinearRegression)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

이미 만들어둔 인스턴스에도 즉시 적용된다. 클래스에 붙는 것이기 때문이다.

## 4. 학습

```python
model = LinearRegression()
trainer = dt.Trainer(max_epochs=20)
trainer.fit(model, data)
```

이 셀의 출력에서 손실 곡선이 **제자리에서 갱신된다.** 에폭마다 새 그림이
쌓이는 게 아니라 하나가 계속 다시 그려진다.

디바이스는 알아서 고른다. Mac이면 `mps`, NVIDIA GPU가 있으면 `cuda`다.

```python
trainer.device
```

```
device(type='mps')
```

`nn.LazyLinear`를 썼는데 수동 초기화를 하지 않았다. `fit()`이 optimizer를
만들기 전에 더미 forward를 한 번 돌려 파라미터를 실체화한다.

에폭별 손실은 `history`에 남는다.

```python
trainer.history["train_loss"][-1], trainer.history["val_loss"][-1]
```

```
(0.00033, 0.00041)
```

## 5. 평가와 체크포인트

학습이 끝난 모델을 검증셋 전체에 돌린다.

```python
p = trainer.predict(data)
p.outputs.shape, len(p)
```

```
(torch.Size([40, 1]), 40)
```

분류였다면 `p.accuracy`, `p.preds`, `p.confidence`를 바로 쓸 수 있다.
회귀는 `p.outputs`를 직접 본다.

저장과 복원:

```python
trainer.save_checkpoint("linreg.pt")

restored = LinearRegression()
restored(data.X[:1])  # LazyLinear 실체화
meta = dt.Trainer.load_checkpoint("linreg.pt", restored)
meta
```

```
{'epoch': 19, 'hparams': {'lr': 0.03}}
```

복원 전에 더미 forward 한 번이 필요하다. `LazyLinear`는 파라미터가 생기기
전까지 `load_state_dict`로 채울 곳이 없다.

## 다음

- 실제 데이터셋을 쓰려면 [데이터](guide/data.md)
- 과적합 전 최적 가중치를 건지려면 [최적 가중치와 조기 종료](guide/best.md)
- 예측을 자세히 뜯어보려면 [사후 평가](guide/evaluate.md)
