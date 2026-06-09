# 어텐션 기반 모듈형 컨셉 보틀넥 모델 (Attention-CBM) 파이프라인

이 프로젝트는 해석가능한 분류 모델을 만들기 위해서 Concept Bottleneck Model을 이용하여 새를 분류하는 프로젝트입니다.

CBM은 이미지에서 바로 라벨을 블랙박스 모델로 예측하는 대신, 중간에 concept bottleneck을 넣어서 사람이 인식할 수 있는 특징을 분류하고, 이를 이용하여 라벨을 예측합니다.

데이터셋으로는 CUB-200-2011을 사용하였다.

---

## 디렉토리 구조
```
project_root/
├── checkpoints/            # 학습된 백본 모델별 가중치 저장소
├── configs/
│   ├── train_config.yaml   # 모델, 옵티마이저, 스케줄러, 얼리스톱 설정 YAML
│   ├── cub_train_config.yaml # CUB LoRA, calibration, CooP 평가 설정
│   └── ...                 # MILK10K, Derm7pt, CheXpert 데이터셋별 설정
├── data/                   # (git 제외) MILK10K, CUB 등 원본 데이터 저장소
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base_dataset.py # 추상 데이터셋 기본 클래스
│   │   └── milk10k.py      # MILK10K 다중 클래스 데이터셋 로더
│   ├── models/
│   │   ├── __init__.py
│   │   └── cbm_factory.py  # UniversalFlexibleCBM 레이아웃 빌더
│   ├── tti/
│   │   ├── common.py       # TTI 공통 metric/logit helper
│   │   └── coop.py         # CooP scoring 및 validation fitting
│   └── utils/
│       ├── __init__.py
│       ├── concept_bias.py # Calibration split 및 concept-bias 학습
│       └── metrics.py      # 정확도 및 평가지표 연산 유틸
├── app.py                  # Gradio 체크포인트 탐색 웹 애플리케이션
├── download_CUB-200-2011.py # CUB 다운로드, 검증, 압축 해제 스크립트
├── eval_cbm.py             # 평가 및 TTI 벤치마크 진입점
├── generate_concept_config.py # 메타데이터 기반 컨셉 설정 추출기
├── main.py                 # 학습 및 평가 통합 진입점 스크립트
├── requirements.txt        # 종속성 파일
└── README_KR.md
```

---

## 환경 설정

본 프로젝트는 종속성 관리 및 빌드 도구로 `uv`를 사용합니다.

```bash
# 종속성 패키지 동기화 및 가상환경 설치
uv sync
```

*(만약 `uv`를 사용하지 않으신다면 `pip install -r requirements.txt` 명령어를 실행하십시오)*

---

## 학습 및 평가

`main.py` 진입점으로 학습 파이프라인을 실행합니다. 모델, 데이터셋, 컨셉, 학습 설정은 `--config_path`로 전달한 YAML 파일에서 읽습니다.

### 1) CUB-200-2011 다운로드 및 학습
CUB archive를 다운로드하고 검증한 뒤 프로젝트용 concept config를 생성합니다:
```bash
# Kaggle에서 다운로드
curl -L -o data/cub2002011.zip\
  https://www.kaggle.com/api/v1/datasets/download/wenewone/cub2002011

# 또는 스크립트 사용 (칼텍 서버)
uv run python download_CUB-200-2011.py --data-dir data

# 학습 이후에 concept 인덱스 생성
uv run python scratch/convert_cub_attributes.py
```

### 2) 기본 CBM 학습 (설정 파일 값 사용)
`configs/train_config.yaml`의 기본 설정으로 CBM을 학습합니다:
```bash
uv run python main.py --config_path configs/cub_train_config.yaml
```

### 8) 평가 및 TTI 벤치마크
`eval_cbm.py`로 표준 CBM 평가와 TTI 벤치마크를 실행합니다. 
```bash
uv run python eval_cbm.py \
    --checkpoint PATH_TO_CHECKPOINT \

# 별도 파일로 저장
TQDM_DISABLE=1 uv run sh -c "python eval_cbm.py --checkpoint PATH_TO_CHECKPOINT
    2>&1 | tee eval.txt"
```

## 재현 가능성

훈련 데이터셋을 다운로드 한 뒤 훈련, 평가를 진행하면 결과를 재현할 수 있습니다. 하지만, 데이터셋이 크고 훈련이 오래 걸리기 때문에 가중치를 포함하였습니다.

RTX 4090 환경에서는 훈련에 약 20분, 평가에 약 1분이 소요되었습니다.

## AI 사용

코드 구현 전반에 코딩 에이전트를 사용하였습니다.
