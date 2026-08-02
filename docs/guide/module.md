# 모델

`dt.Module`은 `nn.Module`에 하이퍼파라미터 저장과 플롯 훅을 얹은 것이다.
PyTorch 모델 작성 방식은 그대로다.

## 채워야 할 세 가지

```python
class MyNet(dt.Module):
    def forward(self, X): ...
    def loss(self, y_hat, y): ...
    def configure_optimizers(self): ...
```

`loss`와 `configure_optimizers`는 기본 구현이 `NotImplementedError`를 던진다.
반드시 채워야 한다.

`forward`는 예외다. **`self.net`을 정의하면 자동으로 위임된다.**

```python
class MyNet(dt.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(256),
                                 nn.ReLU(), nn.LazyLinear(10))
```

`self.net`도 없고 `forward`도 없으면 호출 시점에 막힌다.

```
AssertionError: implement forward() or assign self.net
```

## `save_hyperparameters()` 와 호출 순서

`__init__`의 인자를 전부 인스턴스 속성으로 만들고 `self.hparams`에도 담는다.

```python
class MyNet(dt.Module):
    def __init__(self, lr=0.01, num_hiddens=256):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(10))

model = MyNet(lr=0.1)
model.lr, model.hparams
```

```
(0.1, {'lr': 0.1, 'num_hiddens': 256})
```

`hparams`는 체크포인트에 함께 저장되므로, 나중에 이 모델이 어떤 설정으로
학습됐는지 파일만 보고 알 수 있다.

!!! warning "순서를 지켜야 한다"
    `super().__init__()` **먼저**, `save_hyperparameters()` **나중**이다.

    부모의 `__init__`도 자기 인자로 `hparams`를 덮어쓴다. 순서를 바꾸면
    당신이 넘긴 값이 부모 기본값에 지워진다.

    ```python
    def __init__(self, lr=0.1):
        self.save_hyperparameters()   # 여기서 lr=0.1 저장
        super().__init__()            # 부모가 hparams 를 자기 것으로 덮음
    ```

지역 변수는 잡히지 않는다. 선언된 인자만 읽는다.

```python
def __init__(self, lr=0.1):
    scratch = lr * 2      # hparams 에 안 들어간다
    self.save_hyperparameters()
```

특정 인자를 빼려면 `ignore`를 쓴다.

```python
self.save_hyperparameters(ignore=["api_key"])
```

## `@add_to_class` — 노트북 셀 사이에서 메서드 붙이기

노트북에서 클래스를 고칠 때마다 정의 셀로 돌아가 다시 실행하고, 그러면
아래 셀들도 전부 다시 돌려야 하는 문제를 없앤다.

```python
# 셀 3
class MyNet(dt.Module):
    def __init__(self, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(10)

# 셀 7 — 한참 뒤에
@dt.add_to_class(MyNet)
def loss(self, y_hat, y):
    return F.cross_entropy(y_hat, y)
```

클래스에 붙기 때문에 **이미 만들어둔 인스턴스에도 즉시 적용된다.**

데코레이터는 원래 함수를 그대로 돌려준다. 정의한 셀에서도 그 이름을 계속
쓸 수 있다는 뜻이다.

```python
@dt.add_to_class(MyNet)
def loss(self, y_hat, y):
    return F.cross_entropy(y_hat, y)

loss          # <function loss at 0x...> — None 이 아니다
```

## 배치 규약

`training_step`과 `validation_step`의 기본 구현은 배치를 이렇게 해석한다.

```
batch[:-1]   입력  (여러 개일 수 있다)
batch[-1]    정답
```

`self(*batch[:-1])`로 forward를 부르므로, 입력이 두 개인 모델은
`(X1, X2, y)` 배치를 그대로 받는다.

## 정확도 곡선 추가하기

기본 구현은 손실만 그린다. 정확도도 보고 싶으면 `validation_step`을 덮어쓴다.

```python
@dt.add_to_class(MyNet)
def validation_step(self, batch):
    y_hat = self(*batch[:-1])
    loss = self.loss(y_hat, batch[-1])
    self.plot('loss', loss, train=False)
    self.plot('acc', (y_hat.argmax(-1) == batch[-1]).float().mean(), train=False)
    return loss
```

`plot`으로 찍은 스칼라는 같은 보드에 다른 색·선스타일로 그려진다.
반환값은 여전히 손실이어야 한다. `Trainer`가 이것으로 `history`를 채우고
최저점을 판정하기 때문이다.

!!! note "`plot`은 `history`에 남지 않는다"
    `plot`은 그림만 그린다. `trainer.history`에는 `train_loss`와
    `val_loss`만 쌓인다. 정확도를 숫자로 보관하려면 직접 리스트에 모으거나,
    학습이 끝난 뒤 [`trainer.predict(data).accuracy`](evaluate.md)를 쓴다.

## 다음

- [데이터](data.md) — 모델에 배치를 공급하는 쪽
- [학습기](trainer.md) — `fit`이 이 규약들을 어떻게 쓰는지
