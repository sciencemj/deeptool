# 최적 가중치와 조기 종료

30 에폭을 돌리면 손에 남는 건 **30번째 가중치**다. 검증 손실이 8번째에서
최저였더라도 그 모델은 이미 사라진 뒤다.

두 기능이 그 문제를 나눠 맡는다. 따로 쓰면 각각 반쪽이다.

| 단독일 때 | 문제 |
|---|---|
| 조기 종료만 | 멈추긴 하는데 최저점을 지나 나빠진 가중치가 남는다 |
| best 스냅샷만 | 최적 가중치는 건지는데 쓸모없는 에폭을 끝까지 돈다 |

## 표준 절차

```python
trainer = dt.Trainer(max_epochs=100, patience=5)
trainer.fit(model, data)

len(trainer.history["val_loss"])             # 24 — 100까지 안 감
trainer.best_epoch, trainer.best_val_loss    # (18, 0.2913)

trainer.restore_best()                       # 18 을 반환
```

## `fit()`은 가중치를 안 바꾼다

`restore_best()`를 부르기 전까지 모델은 **마지막 에폭 상태**다.
자동 복원하지 않는 것은 의도다 — 두 시점 성능을 비교할 수 있어야 한다.

```python
trainer.predict(data).accuracy    # 마지막 에폭 기준
trainer.restore_best()
trainer.predict(data).accuracy    # 최저점 기준
```

`restore_best()`는 **모델 가중치만** 되돌린다. optimizer 상태는 그대로다.
목적이 "가장 좋은 모델로 평가·추론"이지 학습 재개가 아니기 때문이다.

## 인자 네 개

```python
dt.Trainer(max_epochs, ...,
           snapshot_best=True, best_path=None,
           best_with_optim=False, patience=None)
```

| 인자 | 기본 | 의미 |
|---|---|---|
| `snapshot_best` | `True` | 스냅샷을 만들 것인가 |
| `best_path` | `None` | `None`이면 메모리, 경로면 파일 |
| `best_with_optim` | `False` | 파일에 optimizer 상태도 넣을 것인가 |
| `patience` | `None` | 몇 에폭 개선이 없으면 멈출 것인가 |

`best_val_loss`와 `best_epoch`는 **`snapshot_best=False`여도 계속 추적된다.**
float 비교라 비용이 없고, 몇 번째가 최저였는지는 그 자체로 쓸모가 있다.
끄면 복사·쓰기만 건너뛴다.

## 손실이 계속 줄기만 할 때

단조 감소하면 매 에폭이 새 최저점이라 매번 스냅샷이 뜬다. 이게 최대 빈도다.

**메모리는 누적되지 않는다.** 새 `deepcopy`가 이전 사본을 대체하고 옛 것은
GC된다. 항상 모델 하나 분량이다.

| 모델 | 파라미터 | 에폭당 복사 |
|---|---|---|
| FashionMNIST MLP (784→256→10) | 20만 | 0.8MB — 무시 가능 |
| ResNet-50 | 2500만 | 100MB, 약 50ms |

에폭 하나가 수십 초인 걸 생각하면 메모리 모드 비용은 없는 것과 같다.
그래서 기본값이 메모리다.

끝까지 단조 감소하면 `best_epoch`가 마지막 에폭이고 `restore_best()`는
아무것도 안 바꾼다. `patience`도 발동하지 않는다.
**그 자체가 "더 오래 학습해야 한다"는 신호다.**

## 파일로 남기기 — optimizer 상태를 왜 빼는가

```python
dt.Trainer(max_epochs=100, patience=5, best_path="best.pt")
```

파일에는 **모델 가중치만** 들어간다. 이유는 산수다.

`restore_best()`는 optimizer 상태를 읽지 않는다. 그런데 Adam은 파라미터당
moment를 2개 들고 있어 optimizer 상태가 **모델 크기의 2배**다.

```
ResNet-50 + Adam, 개선될 때마다 쓰기
  model  100MB
  optim  200MB   ← 한 번도 읽히지 않는다
  ─────────────
         300MB
```

읽지 않을 200MB를 매 에폭 쓰는 건 낭비다. 최저점부터 학습을 재개할 계획이
**있을 때만** 켠다.

```python
dt.Trainer(max_epochs=100, best_path="best.pt", best_with_optim=True)
# 나중에
dt.Trainer.load_checkpoint("best.pt", model, optim)
```

### 쓰다가 중단되면

`best_path`는 개선마다 같은 경로를 덮어쓰는 파일이다. 쓰는 도중 끊기면
그때까지 쌓은 최적 가중치를 통째로 잃는다.

`<path>.tmp`에 먼저 쓰고 `os.replace`로 교체한다. POSIX와 Windows 양쪽에서
원자적이고, 실패하면 이전 파일이 그대로 남는다.

## 조기 종료의 두 가지 방어

`patience=3`은 "개선 없는 에폭이 3번 연속이면 멈춘다"는 뜻이다.
`best_epoch=8`이고 현재가 11이면 `11 - 8 = 3 >= 3`이라 멈춘다.

잘못된 설정은 즉시 막는다.

**`patience=0`은 거부된다.** `epoch - best_epoch >= 0`이 최저점 에폭에서도
참이라 첫 에폭 직후 멈춘다. 의미가 없다.

```
ValueError: patience must be at least 1 (got 0)
```

**검증 데이터 없이 `patience`도 거부된다.** `best_epoch`가 영원히 `None`이라
조기 종료가 절대 발동하지 않는다. 조용히 무시하면 왜 안 멈추는지 알 방법이 없다.

```
ValueError: patience needs validation data.
```

## `restore_best()`가 던지는 세 가지

| 상황 | 메시지 |
|---|---|
| `fit()` 전 호출 | `fit() has not run yet.` |
| 검증 데이터 없음 | `No snapshot: there was no validation data.` |
| `snapshot_best=False` | `No snapshot: trained with snapshot_best=False. (best was epoch 8, val_loss 0.2913)` |

세 번째가 특히 쓸모 있다. 껐다는 사실을 알리면서 최저점 정보는 그대로 준다.
`max_epochs=8`로 다시 돌리면 된다.

## val_loss 기준의 한계

`best_epoch`는 **검증 손실** 최저점이다. 분류에서 이게 정확도 최고점과
같지 않다.

교차엔트로피는 `-log p(정답)`이라 확신도에 연속 반응하고, 정확도는 argmax의
0/1이다. 최저 CE 지점 이후:

- 이미 맞추는 샘플: `p` 0.90 → 0.99. CE 이득 미미 (0.105 → 0.010)
- 틀리는 샘플: `p(정답)` 0.10 → 0.01. CE 페널티 폭발 (2.3 → 4.6)

평균 CE는 오르는데, 결정 경계는 계속 다듬어져 애매한 샘플 몇 개가 정답으로
넘어간다. **정확도 정점이 손실 최저점보다 뒤에 오는 게 보통이다.**

### 먼저 재라

차이가 실제로 있는지부터 확인한다. 대개 노이즈다.

```python
p_last = trainer.predict(data)      # fit 직후 = 마지막 에폭
acc_last, n = p_last.accuracy, len(p_last)

trainer.restore_best()              # 이 뒤로 모델이 바뀐다
acc_best = trainer.predict(data).accuracy

import math
sigma = math.sqrt(acc_best * (1 - acc_best) / n)
print(f"best {acc_best:.4f}   last {acc_last:.4f}")
print(f"diff {(acc_last - acc_best) * 100:+.2f}%p   1σ = {sigma * 100:.2f}%p")
```

순서가 중요하다. `restore_best()`가 모델을 바꾸므로 last를 먼저 잰다.

`n=10000, p≈0.93`이면 1σ가 약 0.25%p다. **차이가 1σ 안이면 어느 쪽을 골라도
같다.** 실제 FashionMNIST MLP로 재보면 0.01%p — 1만 장 중 한 장 차이가
나오기도 한다.

### 차이가 진짜라면

무엇을 고를지는 **다운스트림에서 뭘 쓰느냐**에 달렸다.

| 쓰는 것 | 맞는 체크포인트 |
|---|---|
| argmax 예측만 | 정확도 최고점 |
| `p.confidence`, 확률, 임계값 컷, 앙상블 | **손실 최저점** |

정확도 최고점 모델은 calibration이 나쁘다. 정확히 위에서 설명한 이유로,
틀린 것을 더 확신하는 상태이기 때문이다. 그 모델의 `p.confidence`는 과신이라
"확신도 0.99인데 틀림" 케이스가 늘어난다.

`deeptool`이 손실 최저점을 기준으로 삼는 것은 이쪽이 보수적으로 안전해서다.

## 다음

- [사후 평가](evaluate.md) — `predict`와 `Predictions`
- [학습기](trainer.md) — 학습 루프 전체
