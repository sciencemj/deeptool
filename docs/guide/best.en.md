# Best weights & early stopping

Train for 30 epochs and what you hold is the **30th set of weights**. If
validation loss bottomed out at epoch 8, that model is already gone.

Two features split the problem. Alone, each is half a solution.

| On its own | The problem |
|---|---|
| Early stopping only | It stops, but leaves you past the minimum with worse weights |
| Best snapshot only | It keeps the good weights, but burns through useless epochs |

## The standard recipe

```python
trainer = dt.Trainer(max_epochs=100, patience=5)
trainer.fit(model, data)

len(trainer.history["val_loss"])             # 24 — nowhere near 100
trainer.best_epoch, trainer.best_val_loss    # (18, 0.2913)

trainer.restore_best()                       # returns 18
```

## `fit()` leaves the weights alone

Until you call `restore_best()`, the model holds the **last epoch's** weights.
Not restoring automatically is deliberate — you should be able to compare.

```python
trainer.predict(data).accuracy    # last epoch
trainer.restore_best()
trainer.predict(data).accuracy    # best epoch
```

`restore_best()` restores **model weights only**. Optimizer state is untouched,
because the point is evaluating with the best model, not resuming training.

## The four arguments

```python
dt.Trainer(max_epochs, ...,
           snapshot_best=True, best_path=None,
           best_with_optim=False, patience=None)
```

| Argument | Default | Meaning |
|---|---|---|
| `snapshot_best` | `True` | Whether to snapshot at all |
| `best_path` | `None` | `None` keeps it in memory; a path writes a file |
| `best_with_optim` | `False` | Also store optimizer state in that file |
| `patience` | `None` | Stop after this many epochs without improvement |

`best_val_loss` and `best_epoch` are tracked **even with
`snapshot_best=False`**. Comparing floats costs nothing, and knowing which
epoch was best is useful on its own. Disabling only skips the copy or write.

## When the loss just keeps falling

Under a monotonic decrease every epoch is a new best, so a snapshot happens
every time. That is the maximum frequency.

**Memory does not accumulate.** Each new `deepcopy` replaces the previous one
and the old copy is collected. You always hold one model's worth.

| Model | Parameters | Copy per epoch |
|---|---|---|
| FashionMNIST MLP (784→256→10) | 200K | 0.8MB — negligible |
| ResNet-50 | 25M | 100MB, about 50ms |

Against an epoch that takes tens of seconds, the memory mode costs nothing.
That is why it is the default.

If the loss falls monotonically to the end, `best_epoch` is the final epoch,
`restore_best()` changes nothing and `patience` never fires. **That itself
tells you to train longer.**

## Writing to a file — why optimizer state is excluded

```python
dt.Trainer(max_epochs=100, patience=5, best_path="best.pt")
```

The file holds **model weights only**. The reason is arithmetic.

`restore_best()` never reads optimizer state. Yet Adam keeps two moment tensors
per parameter, making optimizer state **twice the size of the model**.

```
ResNet-50 + Adam, written on every improvement
  model  100MB
  optim  200MB   ← never read back
  ─────────────
         300MB
```

Writing 200MB per epoch that nobody reads is waste. Turn it on only when you
actually plan to resume training from the best epoch.

```python
dt.Trainer(max_epochs=100, best_path="best.pt", best_with_optim=True)
# later
dt.Trainer.load_checkpoint("best.pt", model, optim)
```

### If the write is interrupted

`best_path` is one file overwritten on every improvement. An interrupted write
destroys every good weight collected so far.

The payload goes to `<path>.tmp` first, then `os.replace` swaps it in. That is
atomic on POSIX and Windows alike, and a failure leaves the previous file
intact.

## Two guards on early stopping

`patience=3` means "stop after three consecutive epochs without improvement."
With `best_epoch=8` and the current epoch at 11, `11 - 8 = 3 >= 3` stops it.

Bad configuration is rejected immediately.

**`patience=0` is refused.** `epoch - best_epoch >= 0` is true even at the best
epoch, so it would stop right after the first one. Meaningless.

```
ValueError: patience must be at least 1 (got 0)
```

**`patience` without validation data is refused.** `best_epoch` would stay
`None` forever and early stopping could never fire. Ignoring that silently
would leave you with no way to find out why it never stops.

```
ValueError: patience needs validation data.
```

## What `restore_best()` raises

| Situation | Message |
|---|---|
| Called before `fit()` | `fit() has not run yet.` |
| No validation data | `No snapshot: there was no validation data.` |
| `snapshot_best=False` | `No snapshot: trained with snapshot_best=False. (best was epoch 8, val_loss 0.2913)` |

The third one earns its keep: it tells you snapshotting was off while still
handing over the best-epoch information. Rerun with `max_epochs=8`.

## The limits of selecting on val_loss

`best_epoch` is the minimum of **validation loss**. For classification that is
not the same as peak accuracy.

Cross-entropy is `-log p(correct)`, a continuous response to confidence.
Accuracy is a 0/1 argmax. After the cross-entropy minimum:

- Samples already correct: `p` 0.90 → 0.99. Tiny gain (0.105 → 0.010)
- Samples getting it wrong: `p(correct)` 0.10 → 0.01. Penalty explodes (2.3 → 4.6)

Mean cross-entropy rises, yet the decision boundary keeps sharpening and a few
borderline samples flip to correct. **Peak accuracy usually arrives after the
loss minimum.**

### Measure first

Check whether the difference is real. Usually it is noise.

```python
p_last = trainer.predict(data)      # right after fit = last epoch
acc_last, n = p_last.accuracy, len(p_last)

trainer.restore_best()              # the model changes from here
acc_best = trainer.predict(data).accuracy

import math
sigma = math.sqrt(acc_best * (1 - acc_best) / n)
print(f"best {acc_best:.4f}   last {acc_last:.4f}")
print(f"diff {(acc_last - acc_best) * 100:+.2f}%p   1σ = {sigma * 100:.2f}%p")
```

Order matters: `restore_best()` mutates the model, so measure `last` first.

At `n=10000, p≈0.93`, one sigma is about 0.25%p. **If the gap sits inside one
sigma, either checkpoint is the same choice.** A real FashionMNIST MLP run can
come out at 0.01%p — one image out of ten thousand.

### If the difference is real

Which one to keep depends on **what you do downstream**.

| What you use | The right checkpoint |
|---|---|
| Argmax predictions only | Peak accuracy |
| `p.confidence`, probabilities, threshold cuts, ensembles | **Loss minimum** |

The peak-accuracy model is worse calibrated, for exactly the reason described
above: it has grown more confident about the things it gets wrong. Its
`p.confidence` is inflated, so "99% confident and wrong" cases multiply.

`deeptool` selects on loss because that is the conservative default.

## Next

- [Evaluation](evaluate.md) — `predict` and `Predictions`
- [Trainer](trainer.md) — the whole training loop
