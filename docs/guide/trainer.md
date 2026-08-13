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
 'patience': 3, 'log_dir': None}
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

모델이 `self.log("iou", value)`를 부르면 사용자 지표도 같은 dict에 에폭
평균으로 들어간다. 어떤 에폭에 값이 없으면 위치를 맞추기 위해 `None`이 들어간다.

## 디스크에 학습 기록 남기기

```python
trainer = dt.Trainer(max_epochs=50, plot=False, log_dir="runs/exp1")
trainer.fit(model, data)
```

완료된 에폭마다 JSONL 한 줄을 append하고 즉시 flush한다.

```
runs/exp1/
  meta.json
  history.jsonl
```

`meta.json`에는 시작 시각, 실제 device, 모델 클래스와 Trainer/모델
하이퍼파라미터가 들어간다. `history.jsonl`의 각 행은 다음처럼 자유형이다.

```json
{"epoch": 0, "train_loss": 2.4724, "val_loss": 2.3155, "iou": 0.3248, "lr": 0.001, "sec": 116.2}
```

프로세스가 다음 에폭 중에 죽어도 이미 끝난 줄은 남는다. 일반 스크립트에서는
`log_dir`을 준 경우에만 같은 필드가 에폭당 한 줄로 stdout에도 나온다.
노트북에서는 라이브 보드만 쓰고 이 문장은 출력하지 않는다. `log_dir=None`이면
기존처럼 파일과 에폭 출력이 모두 없다.

여러 실행을 읽고 지표별로 겹쳐 그린다.

```python
runs = dt.load_runs("runs")
figures = dt.plot_runs(runs)
```

`load_runs`는 `{"실행명": {"지표명": [값, ...]}}`를 반환한다. 어떤 모델에만
있는 지표도 그대로 읽으며, 에폭 중간에 없는 값은 `None`으로 정렬한다.
`plot_runs`는 `epoch`을 제외한 지표마다 열린 Matplotlib Figure 하나를 반환한다.

## 학습률 스케줄러

`configure_optimizers()`는 기존처럼 optimizer 하나를 반환하거나
`(optimizer, scheduler)`를 반환한다.

```python
def configure_optimizers(self):
    optim = torch.optim.Adam(self.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optim, step_size=10, gamma=0.1
    )
    return optim, scheduler
```

일반 scheduler의 `step()`은 에폭 학습·검증·기록이 끝난 뒤 한 번 호출된다.
따라서 JSONL의 `lr`은 그 행의 에폭에서 실제로 사용한 값이고, 바뀐 값은 다음
에폭부터 보인다.

`ReduceLROnPlateau`만 `scheduler.step(val_loss)`로 호출한다. 판단할 검증
손실이 필요하므로 검증 dataloader가 없으면 학습 시작 전에 `ValueError`가 난다.
매 배치마다 step해야 하는 `OneCycleLR` 같은 scheduler와 AMP는 아직 지원하지
않는다.

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
5. `model.configure_optimizers()`로 optimizer와 선택적 scheduler를 만든다
6. 선택했다면 실행 메타데이터를 쓴다
7. 에폭 루프 — 학습 → 검증 → 최저점 판정 → 에폭 기록 → scheduler → 조기 종료

에폭 하나(`fit_epoch`)는 학습 배치를 돌며 `training_step`을 부르고,
검증 배치를 `torch.no_grad()` 아래 `validation_step`으로 돌린다.
`model.train()`과 `model.eval()` 전환도 여기서 한다.

## 다음

- [최적 가중치와 조기 종료](best.md) — 7번 단계의 최저점 판정
- [사후 평가](evaluate.md) — 학습이 끝난 뒤
