# Network Attack Detection with Temporal Validation and MLflow

네트워크 Flow 통계 데이터로 `Benign`, `Brute Force`, `DDOS`, `DoS`,
`Infilteration`을 분류한 프로젝트입니다. 대회 점수만 높이는 모델과 실제 IDS에서
운영 가능한 모델의 차이를 분석하고, 시간 누수 점검·Hard Negative Mining·MLflow
실험 관리·설명 가능한 규칙 기반 오탐 억제를 적용했습니다.

> 이 프로젝트는 원본 패킷 payload나 PCAP을 직접 분석하지 않습니다. 패킷에서
> 집계된 Flow Duration, 패킷 길이, Flag 개수, 전송량, 포트 등의 통계값을 사용합니다.

## Problem

- 학습 데이터: 4,638,804 rows
- 테스트 데이터: 1,159,701 rows
- 입력: 41개 원본 네트워크 Flow 컬럼
- 출력: 5개 클래스
- 대회 지표: `Benign`을 제외한 공격 클래스 Weighted F1
- 데이터 라벨의 `Infilteration`은 `Infiltration`의 오타이지만 제출 스키마에 맞춰 유지

`Benign`은 평가 라벨에서 제외되지만 정상 흐름을 공격으로 예측하면 공격 클래스의
Precision이 떨어집니다. 특히 v15에서는 `Benign → Infilteration` 오탐을 줄이는 것이
핵심 과제였습니다.

## Approach

### 1. Leakage-aware temporal validation

- 절대 Timestamp를 제외한 모델과 시간 정보를 활용한 모델을 분리해 해석
- 클래스별 시간순 학습/검증 분할
- 규칙 학습, 규칙 보정, 내부 선택, 최종 홀드아웃 평가 구간 분리
- 테스트 Label 미사용
- 테스트 rolling 피처는 테스트 행이 아닌 과거 학습 이벤트만 참조

### 2. v15 temporal RandomForest

14개 행 단위 Flow 피처와 과거 구간에서 계산한 16개 시간 집계 피처를 결합했습니다.

- 최근 10초·60초 Flow 수
- 최근 패킷·바이트·SYN 수
- Forward/Backward 비율
- 목적지 포트별 Flow 수와 포트 다양성
- 무응답 흐름 비율

RandomForest에 클래스 가중치를 적용하고, 최종 결정에서 `Infilteration` 확률을
`1.225`배 보정했습니다.

### 3. v16–v19 specialist experiments

전체 원본 피처, Hard Negative, 행위 기반 파생 피처, 전용 검증 모델, 확률 혼합을
실험했습니다. 내부 지표가 좋아도 최종 시간 홀드아웃에서 v15를 안정적으로 넘지
못한 후보는 승격하지 않았습니다.

### 4. v20b interpretable rule hybrid

v15가 `Infilteration`이라고 예측한 사례만 대상으로, 차선 클래스가 정답인 영역을
얕은 결정 트리로 탐색했습니다. 최종적으로 선택된 규칙은 다음과 같습니다.

```text
v15 prediction == Infilteration
AND flow_count_60s > 917
AND Pkt Size Avg <= 82.75
→ replace with v15 second-best non-Infilteration class
```

이 규칙은 학습 데이터에 직접 맞추지 않고 별도의 보정 구간에서 Precision을 측정한
뒤, 미사용 시간 홀드아웃에서 최종 평가했습니다.

## Results

| Metric | v15 | v20b |
|---|---:|---:|
| Attack weighted F1 | 0.95656 | **0.95664** |
| Attack macro F1 | 0.79323 | **0.79363** |
| IDS attack precision | 78.98% | **79.17%** |
| IDS attack recall | **97.58%** | **97.58%** |
| IDS attack F1 | 87.30% | **87.41%** |
| Infilteration precision | 10.59% | **10.70%** |
| Benign → Infilteration FP | 4,054 | **4,008** |

최종 홀드아웃에서 규칙은 46건의 Infilteration 예측을 변경했습니다. 46건 모두
오탐이었으며 공격 Recall 손실은 없었습니다. 전체 테스트에서는 1,485건이
`Infilteration → Benign`으로 변경됐지만 테스트 Label은 사용하지 않았으므로 이
변경의 정답 여부는 확정하지 않습니다.

## Repository structure

```text
.
├── elice/
│   └── code.ipynb                     # v20b 전체 재학습·제출 노트북
├── notebooks/
│   ├── code.ipynb                     # v15 대회 제출 노트북
│   └── code_mlflow_pipeline_v*.ipynb  # 버전별 실험 기록
├── scripts/
│   ├── v15_full_refit_submission.py
│   ├── v16_*_validation.py
│   ├── v17*_validation.py
│   ├── v18_*_validation.py
│   ├── v19_*_validation.py
│   └── v20*_validation.py
├── src/                               # 피처·모델·MLflow 공통 코드
├── docs/
│   └── RETROSPECTIVE.md
├── MLFLOW_WORKFLOW.md
└── requirements.txt
```

데이터셋, MLflow 모델, 제출 CSV와 생성된 실험 아티팩트는 라이선스·용량·누수 문제를
방지하기 위해 저장소에서 제외합니다.

## Reproduce the Elice submission

엘리스 환경에서는 대회 규칙에 따라 노트북 이름을 `code.ipynb`로 유지해야 합니다.

1. [`elice/code.ipynb`](elice/code.ipynb)를 엘리스 프로젝트의 `code.ipynb`로 교체합니다.
2. `/mnt/elice/dataset/train.csv`와 `test.csv`가 있는지 확인합니다.
3. 노트북을 처음부터 끝까지 실행합니다.
4. 노트북과 같은 위치에 생성된 `submission.csv`를 제출합니다.

노트북은 `duckdb`와 MLflow 없이도 실행되며, 다음 검사를 수행합니다.

- 전체 학습 데이터 재학습
- 1,159,701개 테스트 ID 순서 보존
- 결측 Label과 중복 ID 검사
- 알려지지 않은 클래스 검사
- v20b 규칙 적용 건수 출력

전체 실행에는 4 CPU 기준 약 3분이 소요됐으며 환경에 따라 달라질 수 있습니다.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

검증 스크립트는 로컬 Parquet 캐시와 MLflow 서버를 사용합니다. 프로젝트 환경에 맞게
데이터 경로와 `MLFLOW_TRACKING_URI`를 설정해야 합니다.

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python scripts/v20b_v15_error_rule_hybrid_validation.py
```

## MLflow setup used in this project

- Tracking backend: PostgreSQL 16
- Artifact store: local filesystem
- Model flavor: MLflow sklearn + python function
- Serialization: cloudpickle
- Logged items: parameters, metrics, tags, model signatures, notebooks, reports and hashes

실제 접속 정보와 비밀번호는 저장소에 포함하지 않습니다.

## Key lessons

- 대회 최적화와 실제 IDS 일반화는 목표가 다를 수 있습니다.
- Timestamp를 무조건 포함하거나 제거하지 않고 사용 목적과 검증 결과로 결정해야 합니다.
- 확인할 수 없는 미래를 과도하게 가정하기보다 누수 없는 검증 구조를 먼저 설계합니다.
- 높은 Recall만으로는 좋은 IDS가 되지 않으며 정상 오경보를 함께 관리해야 합니다.
- 복잡한 모델보다 검증된 단순 규칙이 더 안정적인 개선을 만들 수 있습니다.
- 실패한 실험도 MLflow에 남겨 선택 편향을 줄입니다.

자세한 회고는 [`docs/RETROSPECTIVE.md`](docs/RETROSPECTIVE.md)를 참고하세요.
