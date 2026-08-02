# 사후 평가

학습이 끝난 모델을 데이터셋 전체에 돌려 **샘플별 예측을 한 번에** 모은다.

```python
p = trainer.predict(data)
p
```

```
<Predictions n=10000 outputs=(10000, 10) accuracy=0.9306>
```

기본은 검증셋이다. 학습셋을 보려면 `train=True`.

```python
trainer.predict(data, train=True)
```

`Trainer` 없이 모델과 dataloader만으로도 부를 수 있다.

```python
p = dt.predict(model, data.val_dataloader())
```

## `Predictions` 속성

원본 `outputs`와 `targets`만 저장하고 나머지는 파생한다. 캐시하지 않는다 —
무효화 규칙이 생기는 것보다 매번 계산하는 편이 싸다.

| 속성 | 내용 |
|---|---|
| `outputs` | 모델 원본 출력 `(N, C)` |
| `targets` | 정답 `(N,)` |
| `inputs` | 입력. `keep_inputs=True`일 때만, 아니면 `None` |
| `preds` | `outputs.argmax(dim=-1)` |
| `probs` | `outputs.softmax(dim=-1)` |
| `confidence` | 예측 클래스에 준 확률 |
| `correct` | `preds == targets` (bool 텐서) |
| `accuracy` | 맞은 비율 (float) |
| `len(p)` | 샘플 수 |

`preds`부터 `accuracy`까지는 **분류 전용**이다. 회귀 모델이면 `outputs`를
직접 쓴다.

## 결과는 항상 CPU다

배치를 가속기에서 추론하되 결과는 즉시 CPU로 회수한다. 이유가 두 가지다.

**메모리.** 회수하지 않으면 검증셋 전체가 GPU/MPS 메모리에 쌓인다.
10,000개 로짓은 작지만 `keep_inputs=True`면 이미지까지 올라간다.

**호환.** matplotlib, numpy, sklearn 등 downstream 코드는 CPU 텐서를 기대한다.
매번 `.cpu()`를 붙이게 하는 건 불필요한 마찰이다.

```python
p.outputs.device, p.targets.device
```

```
(device(type='cpu'), device(type='cpu'))
```

## `keep_inputs`는 왜 기본이 꺼져 있나

입력 텐서가 출력보다 훨씬 크다.

```
28×28 이미지 10,000장   31MB
10클래스 로짓 10,000개   400KB
```

**80배 차이다.** 정확도만 볼 거면 이미지를 모을 이유가 없다.
시각화가 필요할 때만 켠다.

```python
p = trainer.predict(data, keep_inputs=True)
p.inputs.shape
```

```
torch.Size([10000, 1, 28, 28])
```

## 틀린 샘플 들여다보기

`correct`로 바로 가른다.

```python
p = trainer.predict(data, keep_inputs=True)

wrong_imgs = p.inputs[~p.correct]
wrong_conf = p.confidence[~p.correct]
wrong_pred = p.preds[~p.correct]
wrong_true = p.targets[~p.correct]

print(f"{(~p.correct).sum()}개 틀림, 평균 확신도 {wrong_conf.mean():.1%}")
```

```
694개 틀림, 평균 확신도 71.3%
```

확신했는데 틀린 것부터 보는 게 보통 유익하다.

```python
order = wrong_conf.argsort(descending=True)
worst = order[:5]                      # 가장 확신하고 틀린 5개
```

## 분류 전용 속성의 안전장치

회귀 모델의 출력은 `(N, 1)`이다. 여기에 `accuracy`를 부르면 `argmax`가
`(N,)`을 만들고 `(N, 1)`과 비교되며 **브로드캐스트로 `(N, N)` 텐서가**
생긴다. N=10,000이면 100MB짜리에 의미 없는 숫자가 나온다.

그래서 shape이 안 맞으면 막는다.

```
ValueError: preds shape (4,) does not match targets shape (4, 1).
This is a classification-only property; use outputs directly for regression.
```

`repr`도 마찬가지로 shape이 맞을 때만 정확도를 붙인다.

```python
dt.Predictions(torch.randn(4, 1), torch.randn(4, 1))
```

```
<Predictions n=4 outputs=(4, 1)>
```

## 라이브러리가 하지 않는 것

이미지 그리드, 정규화 되돌리기, 클래스 이름 매핑, 혼동 행렬 — 전부
당신 코드다. 이미지 분류 전용이거나 `torchvision` 의존성을 부르기 때문이다.

`deeptool`은 예측을 모으는 데까지만 한다. 그 위는 자유롭게 짜면 된다.

```python
FASHION_CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for ax, i in zip(axes, worst):
    ax.imshow(wrong_imgs[i].squeeze(0), cmap="gray")
    ax.set_title(f"{FASHION_CLASSES[wrong_true[i]]}\n"
                 f"→ {FASHION_CLASSES[wrong_pred[i]]} ({wrong_conf[i]:.0%})")
    ax.axis("off")
```

## 다음

- [최적 가중치와 조기 종료](best.md) — 어느 시점 모델을 평가할지
- [모델](module.md) — 학습 중에 정확도 곡선 그리기
