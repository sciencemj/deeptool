# 학습기

설정은 전부 `Trainer()` 생성자에 모인다. `fit()`은 인자가 둘뿐이다.

```python
trainer = dt.Trainer(max_epochs=20)
trainer.fit(model, data)
```

trainer 하나가 **하나의 학습 설정**을 뜻한다. 그래서 `trainer.hparams`에
그 설정이 통째로 남는다.

```python
dt.Trainer(max_epochs=20, patience=3).hparams
```

```
{'max_epochs': 20, 'device': None, 'gradient_clip_val': 0, 'plot': True,
 'snapshot_best': True, 'best_path': None, 'best_with_optim': False,
 'patience': 3}
```

## 디바이스 자동 선택

`cuda` → `mps` → `cpu` 순으로 사용 가능한 첫 번째를 고른다.
배치는 학습 루프가 알아서 옮긴다.

확인하는 방법이 세 가지 있고, 각각 뜻이 다르다.

```python
dt.default_device()              # 무엇을 고를지 미리 (Trainer 없이)
trainer.device                   # 이 trainer 가 실제로 쓰는 것
next(model.parameters()).device  # 학습 후 모델이 진짜 올라간 곳
```

강제 지정:

```python
dt.Trainer(max_epochs=20, device="cpu")
```

!!! warning "`hparams['device']`는 자동선택 결과가 아니다"
    ```python
    trainer = dt.Trainer(max_epochs=20)
    trainer.hparams['device']   # None
    trainer.device              # device(type='mps')
    ```

    `hparams`는 생성자에 **넘긴 원본 인자**를 저장한다. 아무것도 안 넘겼으면
    `None`이다. 실제로 쓰이는 디바이스는 `trainer.device`를 봐라.

## `history`

에폭별 평균 손실이 쌓인다.

```python
trainer.history
```

```
{'train_loss': [0.62, 0.48, 0.41, ...], 'val_loss': [0.58, 0.45, 0.43, ...]}
```

조기 종료가 걸렸는지는 길이로 안다.

```python
len(trainer.history["train_loss"]) < trainer.max_epochs   # True 면 일찍 멈춤
```

검증 데이터가 없으면 `val_loss`는 빈 리스트로 남는다.

## LazyLinear 자동 실체화

`nn.LazyLinear`, `nn.LazyConv2d`는 첫 forward 전까지 파라미터가 없다.
그 상태로 `configure_optimizers()`를 부르면 optimizer 생성이 실패한다.

`fit()`은 optimizer를 만들기 **전에** 학습 배치 하나로 `torch.no_grad()`
아래 더미 forward를 돌린다. 그래서 수동 초기화가 필요 없다.

```python
class MyNet(dt.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.LazyLinear(10)   # 입력 크기를 안 적어도 된다

trainer.fit(MyNet(), data)             # 그냥 돌아간다
```

!!! note "체크포인트 복원은 예외다"
    `load_checkpoint`로 복원할 때는 `fit()`을 거치지 않으므로 직접 한 번
    돌려야 한다.

    ```python
    restored = MyNet()
    restored(data.X[:1])               # 여기서 파라미터가 생긴다
    dt.Trainer.load_checkpoint("ckpt.pt", restored)
    ```

## 그래디언트 클리핑

```python
dt.Trainer(max_epochs=20, gradient_clip_val=1.0)
```

`backward()` 후 `optim.step()` 전에 `clip_grad_norm_`을 건다.
0이면(기본값) 아무것도 안 한다.

## 체크포인트

```python
trainer.save_checkpoint("ckpt.pt")
```

파일에 `model`, `optim`, `epoch`, `hparams` 넷이 들어간다.

복원은 정적 메서드라 trainer 없이도 부를 수 있다.

```python
model = MyNet()
model(data.X[:1])                                    # LazyLinear 실체화

# 추론용 — 가중치만
meta = dt.Trainer.load_checkpoint("ckpt.pt", model)

# 학습 재개용 — optimizer 상태까지
optim = model.configure_optimizers()
meta = dt.Trainer.load_checkpoint("ckpt.pt", model, optim)

meta
```

```
{'epoch': 19, 'hparams': {'lr': 0.03}}
```

`optim`을 주느냐로 두 용도가 갈린다.

!!! danger "신뢰할 수 있는 파일만 로드하라"
    `hparams`에 임의의 파이썬 객체가 들어갈 수 있어 `weights_only=False`로
    읽는다. 출처를 모르는 체크포인트는 열지 마라.

## 학습 루프 뜯어보기

`fit()`이 하는 일 순서:

1. `data`에서 dataloader 두 개를 받고 배치 수를 센다
2. `patience`를 썼는데 검증 데이터가 없으면 여기서 막는다
3. `model.trainer`와 `model.board`를 주입한다
4. 더미 forward로 lazy 파라미터를 실체화한다
5. `model.configure_optimizers()`로 optimizer를 만든다
6. 에폭 루프 — 학습 → 검증 → 최저점 판정 → 조기 종료 검사

에폭 하나(`fit_epoch`)는 학습 배치를 돌며 `training_step`을 부르고,
검증 배치를 `torch.no_grad()` 아래 `validation_step`으로 돌린다.
`model.train()`과 `model.eval()` 전환도 여기서 한다.

## 다음

- [최적 가중치와 조기 종료](best.md) — 6번 단계의 최저점 판정
- [사후 평가](evaluate.md) — 학습이 끝난 뒤
