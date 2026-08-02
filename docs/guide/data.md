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
