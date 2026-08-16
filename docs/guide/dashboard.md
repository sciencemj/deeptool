# 실행 대시보드

`history.jsonl`을 읽어 학습 중인 실행과 완료된 실행을 같은 화면에서 비교한다.
Streamlit과 Altair는 선택 의존성이므로 코어 설치 크기는 바뀌지 않는다.

## 설치와 로컬 실행

```bash
uv add "deeptool[dashboard]"
deeptool-dashboard runs/
```

기본 주소는 `http://127.0.0.1:8501`이고 2초마다 디스크를 다시 읽는다. 자동
갱신을 끄려면 다음처럼 실행한다.

```bash
deeptool-dashboard runs/ --refresh 0
```

대시보드는 run root 바로 아래의 `history.jsonl`을 찾는다.

```text
runs/
  exp1/history.jsonl
  exp1/meta.json
  exp2/history.jsonl
```

학습기가 각 행을 flush하므로 실행 중에도 최신 완료 지점까지 보인다. 쓰는 중인
마지막 JSONL 조각만 다음 갱신까지 건너뛰며, 줄바꿈까지 끝난 잘못된 JSON은 해당
실행의 오류로 표시한다. 다른 정상 실행의 차트는 계속 나온다.

## 화면 읽기

- 왼쪽의 실행별 토글로 각 곡선을 독립적으로 켜고 끈다.
- `Focus run`은 상단 KPI, 최신 행, `meta.json` 패널의 기준 실행이다.
- metric 토글로 차트 종류를 고른다.
- `train_loss`와 `val_loss`는 같은 loss 차트에 표시한다. 실행은 색으로,
  train/validation은 실선/점선으로 구분한다.
- IoU, AP50처럼 다른 스칼라는 이름을 해석하지 않고 각자 차트가 된다.
- epoch 실행과 step 실행은 서로 다른 진행 축과 차트 그룹에 표시한다.

표시되는 시각은 추정한 학습 상태가 아니라 `history.jsonl`의 마지막 수정
시각이다. 대시보드는 파일을 읽기만 하며 학습을 제어하지 않는다.

## SSH 원격 학습 보기

가장 안전한 기본 경로는 SSH local forwarding이다. 로컬 터미널에서:

```bash
ssh -L 8501:127.0.0.1:8501 user@training-host
```

그 SSH 세션의 원격 호스트에서:

```bash
deeptool-dashboard runs/ --no-browser
```

로컬 브라우저에서 `http://127.0.0.1:8501`을 연다. Streamlit은 원격
loopback에만 열리고 SSH가 암호화된 통로를 만든다. 로컬 8501 포트를 이미 쓰는
경우 양쪽 포트를 함께 바꾼다.

```bash
ssh -L 9000:127.0.0.1:9000 user@training-host
# remote
deeptool-dashboard runs/ --port 9000 --no-browser
```

## 네트워크에 직접 열기

```bash
deeptool-dashboard runs/ --host 0.0.0.0 --port 8501 --no-browser
```

!!! danger "인증과 TLS가 없다"
    deeptool은 대시보드 인증이나 TLS를 제공하지 않는다. `0.0.0.0`으로 열면
    해당 포트에 도달할 수 있는 네트워크 사용자가 실행 경로, 메타데이터, 학습
    지표를 볼 수 있다. 방화벽과 신뢰할 수 있는 사설망을 직접 구성하지 않았다면
    SSH tunnel을 사용하라.

## 다음

- [학습기](trainer.md) — JSONL 기록과 step/epoch 학습
- [최적 가중치와 조기 종료](best.md) — Focus 지표와 함께 볼 monitor
