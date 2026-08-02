# 데이터

`dt.DataModule`은 학습·검증 dataloader를 한 객체로 묶는다.
서브클래스가 구현할 것은 **`get_dataloader(train)` 하나뿐**이다.

```python
class MyData(dt.DataModule):
    def get_dataloader(self, train):
        ...
```

`train_dataloader()`와 `val_dataloader()`는 각각 `train=True`/`False`로
이것을 부르는 얇은 래퍼다.

## 텐서로 시작할 때

이미 메모리에 텐서가 있으면 `get_tensorloader`가 `TensorDataset`과
`DataLoader` 조립을 대신한다. `train=True`면 셔플한다.

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
```

세 번째 인자가 학습·검증을 가르는 슬라이스다. 모든 텐서에 같은 슬라이스가
적용되므로 X와 y가 어긋날 일이 없다.

## 호출 순서

!!! warning "`super().__init__()` 먼저, `save_hyperparameters()` 나중"
    `DataModule.__init__`은 `root`, `num_workers`, `batch_size` 기본값을
    `hparams`에 넣는다. 순서를 바꾸면 당신이 넘긴 `batch_size`가
    부모 기본값 32로 덮인다.

    ```python
    def __init__(self, batch_size=256):
        super().__init__()            # batch_size=32 (부모 기본값)
        self.save_hyperparameters()   # batch_size=256 으로 덮어씀
    ```

## torchvision 데이터셋 쓰기

`torchvision`은 `deeptool`의 의존성이 아니다. 데이터셋을 끼워 팔지 않는다는
설계 결정이고, 필요한 사람만 설치하면 된다.

```bash
uv run --with torchvision jupyter lab
```

`get_tensorloader`는 텐서 묶음용이므로 여기선 쓰지 않는다.
torchvision `Dataset`을 `DataLoader`로 직접 감싼다.

```python
import torchvision
from torch.utils import data
from torchvision import transforms


class FashionMNIST(dt.DataModule):
    def __init__(self, batch_size=256, root="./data"):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.ToTensor()
        self.train_ds = torchvision.datasets.FashionMNIST(
            root=root, train=True, transform=trans, download=True)
        self.val_ds = torchvision.datasets.FashionMNIST(
            root=root, train=False, transform=trans, download=True)

    def get_dataloader(self, train):
        ds = self.train_ds if train else self.val_ds
        return data.DataLoader(ds, self.batch_size, shuffle=train,
                               num_workers=self.num_workers)
```

확인:

```python
d = FashionMNIST()
X, y = next(iter(d.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
print(len(d.train_dataloader()), len(d.val_dataloader()))
```

```
torch.Size([256, 1, 28, 28]) torch.float32 torch.Size([256]) torch.int64
235 40
```

## Hugging Face 데이터셋

`datasets`도 `deeptool`의 의존성이 아니다. 필요할 때만 설치한다.

```bash
uv run --with datasets jupyter lab
```

`deeptool`의 배치 규약이 이미지와 텍스트를 모두 받아낸다. 손댈 것은
`collate` 하나뿐이다.

```
이미지  batch = (image, label)                     → self(*batch[:-1]) = self(image)
텍스트  batch = (input_ids, attention_mask, label) → self(*batch[:-1]) = self(ids, mask)
```

### HF 가 주는 두 가지 함정

**dict 를 내놓는다.** HF 데이터셋은 `{"image": …, "label": …}` 형태다.
`deeptool`은 `batch[:-1]`을 입력으로, `batch[-1]`을 정답으로 읽으므로 그대로는
안 된다.

```
KeyError: slice(None, -1, None)
```

**이미지가 uint8 0-255 다.** `with_format("torch")`는 텐서로 바꿔줄 뿐
정규화하지 않는다. torchvision `ToTensor()`가 하던 `/255`와 float 변환이 없다.

```
RuntimeError: mat1 and mat2 must have the same dtype, but got Byte and Float
```

이름 때문에 특히 놓치기 쉽다. shape은 `(N, 1, 28, 28)`로 이미 channel-first라
그건 손댈 필요가 없다.

### 이미지

```python
import torch
from torch.utils.data import DataLoader, default_collate
from datasets import load_dataset


class HFMnist(dt.DataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.save_hyperparameters()
        d = load_dataset("ylecun/mnist")
        self.splits = {True: d["train"].with_format("torch"),
                       False: d["test"].with_format("torch")}

    def get_dataloader(self, train):
        def collate(examples):
            b = default_collate(examples)
            return b["image"].float() / 255, b["label"]
        return DataLoader(self.splits[train], self.batch_size,
                          shuffle=train, collate_fn=collate)
```

`collate`가 두 함정을 한 번에 해결한다 — dict를 튜플로 바꾸고 `/255`로
정규화한다. 그 뒤로는 평소와 같다.

```python
data = HFMnist()
trainer = dt.Trainer(max_epochs=5, patience=3)
trainer.fit(model, data)          # model 은 dt.Module 서브클래스
trainer.restore_best()
trainer.predict(data).accuracy
```

### 텍스트

토크나이저가 필요하다. 이것도 `deeptool`의 의존성이 아니다.

```bash
uv run --with datasets --with transformers jupyter lab
```

```python
from transformers import AutoTokenizer

TOK = AutoTokenizer.from_pretrained("distilbert-base-uncased")


class IMDB(dt.DataModule):
    def __init__(self, batch_size=16, max_length=128):
        super().__init__()
        self.save_hyperparameters()
        d = load_dataset("stanfordnlp/imdb")
        self.splits = {True: d["train"].shuffle(seed=0),
                       False: d["test"].shuffle(seed=0)}

    def get_dataloader(self, train):
        def collate(examples):
            enc = TOK([e["text"] for e in examples], truncation=True,
                      padding="max_length", max_length=self.max_length,
                      return_tensors="pt")
            labels = torch.tensor([e["label"] for e in examples])
            return enc["input_ids"], enc["attention_mask"], labels
        return DataLoader(self.splits[train], self.batch_size,
                          shuffle=train, collate_fn=collate)
```

여기서 `with_format("torch")`를 쓰지 않는다. 문자열은 어차피 텐서가 안 되고,
토크나이저가 원문 `str`을 받아야 하기 때문이다.

배치가 3원소가 되지만 `deeptool`은 그대로 받는다. `batch[:-1]`이
`(input_ids, attention_mask)`이므로 `forward`가 인자 둘을 받으면 된다.

```python
class TextNet(dt.Module):
    def __init__(self, vocab_size=TOK.vocab_size, dim=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=TOK.pad_token_id)
        self.head = nn.Linear(dim, 2)

    def forward(self, input_ids, attention_mask):
        h = self.emb(input_ids) * attention_mask.unsqueeze(-1)
        return self.head(h.sum(1) / attention_mask.sum(1, keepdim=True))
```

### 형식은 균일하지 않다

여기까지 읽고 "HF 데이터셋은 다 이렇겠지" 하면 안 된다. **같은 것은 dict
껍데기 하나뿐이다.**

| 데이터셋 | `with_format("torch")` 후 | `default_collate` 후 |
|---|---|---|
| `ylecun/mnist` | `Tensor`, `Tensor` | `(4,1,28,28) uint8`, `(4,) int64` |
| `stanfordnlp/imdb` | **`str`**, `Tensor` | **`list`**, `(4,) int64` |
| `rajpurkar/squad` | `str`×4, 중첩 `dict` | `list`×4, 중첩 `dict` |
| `Helsinki-NLP/opus_books` | `str`, 중첩 `dict` | `list`, 중첩 `dict` |

`with_format("torch")`는 **문자열을 텐서로 바꾸지 못한다.** 그대로 `str`로
남긴다. HF의 한계가 아니라, 토크나이저가 모델마다 달라 데이터셋 단에서 정할 수
없기 때문이다.

QA와 번역은 중첩 dict라 `default_collate`로도 텐서가 되지 않는다. 이런 태스크는
전처리 파이프라인이 본체이고 데이터로더는 마지막 단계다. `deeptool`이 대신해줄
수 있는 부분이 아니다.

!!! note "왜 `HFDataModule`을 만들지 않았나"
    `datasets`는 설치하면 35개, `transformers`는 46개 패키지를 끌고 온다.
    런타임 의존성을 `torch`·`matplotlib`·`ipython` 셋으로 유지하는 것이 이
    라이브러리의 방침이고, 절감되는 코드는 이미지 기준 6줄이다.

    모달리티마다 `collate`가 완전히 달라 하나로 묶이지도 않는다.

위 두 예제는 2026-08-03에 실행해 확인했다. MNIST 4,000/1,000 샘플에 선형
모델 5 epoch로 정확도 0.857, IMDB 800/400 샘플에 embedding + mean-pool
6 epoch로 0.6575가 나왔다. 두 경우 모두 `predict`·`restore_best`·`patience`가
수정 없이 동작했다.

## 두 가지 함정

**이미지 배치는 `(N, 1, 28, 28)`이다.** 4차원이므로 모델 첫 층에
`nn.Flatten()`이 필요하다. 없으면 shape 에러가 난다.

```python
self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(256),
                         nn.ReLU(), nn.LazyLinear(10))
```

**`num_workers` 기본값이 0인 이유.** 노트북에서 0보다 크게 두면 macOS의
프로세스 시작 방식 때문에 멈추는 일이 있다. 스크립트로 돌릴 때만 올린다.

## 검증 데이터가 없을 때

`get_dataloader(train=False)`가 `None`을 돌려주면 검증을 건너뛴다.

```python
def get_dataloader(self, train):
    if not train:
        return None
    return self.get_tensorloader((self.X, self.y), train)
```

이 경우 `history["val_loss"]`는 비어 있고,
[`restore_best()`](best.md)와 `patience`는 쓸 수 없다.
둘 다 검증 손실을 기준으로 삼기 때문이다.

## 다음

- [모델](module.md) — 이 배치를 받는 쪽
- [학습기](trainer.md) — dataloader 를 언제 어떻게 부르는지
