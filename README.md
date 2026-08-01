# ood

주피터 노트북에서 PyTorch 모델을 객체지향으로 다루기 위한 얇은 보조 라이브러리.

모델은 유저가 PyTorch로 직접 작성한다. 이 라이브러리는 그 주변만 담당한다 —
하이퍼파라미터 자동 저장, 셀 간 메서드 추가, 학습 중 손실 곡선 라이브 렌더링,
디바이스 자동 선택, 체크포인트.

## 설치

```bash
uv sync
```

## 라이센스

MIT. `LICENSE` 참고.

설계는 [d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en)의 `d2l/torch.py`를 참고했다.
해당 샘플 코드는 modified MIT(`LICENSE-SAMPLECODE`)로 배포된다.
