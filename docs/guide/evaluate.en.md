# Evaluation

Run a trained model over a whole dataset and collect **per-sample predictions
in one pass**.

```python
p = trainer.predict(data)
p
```

```
<Predictions n=10000 outputs=(10000, 10) accuracy=0.9306>
```

Validation is the default. For the training split, pass `train=True`.

```python
trainer.predict(data, train=True)
```

It also works without a `Trainer`, given a model and a dataloader.

```python
p = dt.predict(model, data.val_dataloader())
```

## `Predictions` attributes

Only the raw `outputs` and `targets` are stored; the rest are derived. Nothing
is cached — recomputing is cheaper than owning an invalidation rule.

| Attribute | Contents |
|---|---|
| `outputs` | Raw model output, `(N, C)` |
| `targets` | Ground truth, `(N,)` |
| `inputs` | Inputs, only with `keep_inputs=True`, otherwise `None` |
| `preds` | `outputs.argmax(dim=-1)` |
| `probs` | `outputs.softmax(dim=-1)` |
| `confidence` | Probability given to the predicted class |
| `correct` | `preds == targets` as a bool tensor |
| `accuracy` | Fraction correct, a float |
| `len(p)` | Number of samples |

Everything from `preds` through `accuracy` is **classification only**. For a
regression model, read `outputs` directly.

## Results are always on CPU

Inference runs on the accelerator, but results come straight back to CPU. Two
reasons.

**Memory.** Otherwise the entire validation set piles up in GPU or MPS memory.
Ten thousand logit vectors are small, but with `keep_inputs=True` the images
come along too.

**Compatibility.** matplotlib, numpy and scikit-learn all expect CPU tensors.
Making you write `.cpu()` every time is friction with no upside.

```python
p.outputs.device, p.targets.device
```

```
(device(type='cpu'), device(type='cpu'))
```

## Why `keep_inputs` is off by default

Input tensors dwarf outputs.

```
10,000 images at 28×28    31MB
10,000 ten-class logits   400KB
```

**Eighty times the size.** If you only want accuracy there is no reason to
collect images. Turn it on when you need to visualize.

```python
p = trainer.predict(data, keep_inputs=True)
p.inputs.shape
```

```
torch.Size([10000, 1, 28, 28])
```

## Inspecting the mistakes

`correct` splits them straight away.

```python
p = trainer.predict(data, keep_inputs=True)

wrong_imgs = p.inputs[~p.correct]
wrong_conf = p.confidence[~p.correct]
wrong_pred = p.preds[~p.correct]
wrong_true = p.targets[~p.correct]

print(f"{(~p.correct).sum()} wrong, mean confidence {wrong_conf.mean():.1%}")
```

```
694 wrong, mean confidence 71.3%
```

Starting with the confident mistakes is usually the most informative.

```python
order = wrong_conf.argsort(descending=True)
worst = order[:5]                      # five most confident errors
```

## The guard on classification-only attributes

A regression model outputs `(N, 1)`. Ask for `accuracy` and `argmax` produces
`(N,)`, which compared against `(N, 1)` **broadcasts into an `(N, N)` tensor**.
At N=10,000 that is 100MB of meaningless numbers.

So a shape mismatch is refused.

```
ValueError: preds shape (4,) does not match targets shape (4, 1).
This is a classification-only property; use outputs directly for regression.
```

`repr` applies the same rule and only shows accuracy when the shapes agree.

```python
dt.Predictions(torch.randn(4, 1), torch.randn(4, 1))
```

```
<Predictions n=4 outputs=(4, 1)>
```

## What the library leaves to you

Image grids, undoing normalization, class-name mapping, confusion matrices —
all yours. They are either image-classification specific or would pull in
`torchvision`.

`deeptool` stops at collecting the predictions. Everything above that is free
for you to shape.

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

## Next

- [Best weights & early stopping](best.md) — which epoch's model to evaluate
- [Model](module.md) — plotting accuracy during training
