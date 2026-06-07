# 어텐션 기반 모듈형 컨셉 보틀넥 모델 (Attention-CBM) 파이프라인

본 프로젝트는 의료 영상 분석 및 미세 시각 분류(Fine-grained visual categorization)를 위한 모듈형 **어텐션 기반 컨셉 보틀넥 모델(Attention-CBM)** 파이프라인입니다.

---

## 주요 기능

- **어텐션 기반 컨셉 예측:** CNN 백본은 공간 feature map을 유지하도록 `global_pool=''`로 구성하고, ViT/DINOv2/ConvNeXt 백본은 patch 또는 token feature를 유지합니다. 컨셉 head는 GAP로 압축된 이미지 벡터가 아니라 이러한 공간/패치 feature에서 컨셉 logit과 attention map을 계산합니다.
- **설정 기반 데이터셋 및 컨셉 구성:** YAML 파일과 `concept_config.json`으로 데이터셋, target class, 범주형 컨셉 그룹, 수치형 컨셉 범위, 학습 기본값을 정의합니다. MILK10K, CUB-200-2011, Derm7pt, CheXpert 설정을 포함합니다.
- **백본 학습 모드:** `--backbone_train_mode {frozen,lora,full}`로 백본을 동결하거나, LoRA로 조정하거나, 전체 미세조정할 수 있습니다. LoRA는 ViT/DINOv2 계열 백본에서 지원되며 체크포인트 로드 시 호환되는 모드를 감지합니다.
- **학습 워크플로 지원:** 다단계 학습은 `--resume_checkpoint`와 `--save_filename`, phase별 early stopping, 불균형 target을 위한 inverse-frequency class weight, bottleneck activation dropout, 선택적 사후 concept bias calibration을 지원합니다.
- **평가 및 테스트 시점 개입(TTI):** `eval_cbm.py`는 Accuracy, Macro-F1, Macro-F2 기준으로 concept, target, **Classification (GT Concept)** 지표를 출력합니다. 범주형 컨셉 그룹은 argmax one-hot 예측으로 평가하고 singleton 컨셉은 `logit > 0` 규칙으로 평가하며, TTI 벤치마크는 group-level, concept-level, uncertainty-based, CooP policy를 포함합니다.
- **Gradio 인터랙티브 탐색기:** 앱은 체크포인트를 로드하고, 그룹화된 컨셉별 attention map을 보여주며, 범주형 개입은 dropdown으로 수치형 개입은 실제 단위 slider로 제공합니다. 범주형 컨셉의 top-1/top-2 probability margin을 기준으로 불확실한 컨트롤을 열 수 있습니다.

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
# uv 초기화 (필요한 경우)
uv init --python 3.12

# 종속성 패키지 동기화 및 가상환경 설치
uv sync
```

*(만약 `uv`를 사용하지 않으신다면 `pip install -r requirements.txt` 명령어를 실행하십시오)*

---

## 주요 사용법 및 명령어 예시

`main.py` 진입점으로 학습 파이프라인을 실행합니다. 모델, 데이터셋, 컨셉, 학습 설정은 `--config_path`로 전달한 YAML 파일에서 읽습니다.

### 1) 기본 CBM 학습 (설정 파일 값 사용)
`configs/train_config.yaml`의 기본 설정으로 CBM을 학습합니다:
```bash
uv run python main.py --config_path configs/train_config.yaml
```

### 2) CLI 파라미터 덮어쓰기 (Overrides)
필요한 학습 파라미터만 명령행에서 덮어씁니다:
```bash
uv run python main.py \
    --config_path configs/train_config.yaml \
    --epochs 15 \
    --batch_size 32 \
    --lambda_c 3.0 \
    --lr 0.0005
```

### 3) 백본 학습 모드
`--backbone_train_mode`로 백본 학습 방식을 제어합니다. `frozen`은 사전학습 feature 보존, `lora`는 ViT/DINOv2 기반 parameter-efficient adaptation, `full`은 전체 백본 미세조정에 사용합니다:
```bash
uv run python main.py --config_path configs/train_config.yaml --backbone_train_mode frozen
uv run python main.py --config_path configs/cub_train_config.yaml --backbone_train_mode lora
uv run python main.py --config_path configs/train_config.yaml --backbone_train_mode full
```

### 4) 분류기 헤드 동결 (Classifier Head Freeze)
최종 분류기 헤드를 고정하고 백본과 컨셉 projection layer를 업데이트합니다:
```bash
uv run python main.py --config_path configs/train_config.yaml --freeze_head
```

### 5) 다단계 학습 (Multi-Stage Training)
`--save_filename` 및 `--resume_checkpoint` 옵션으로 여러 학습 단계를 순서대로 연결합니다.
- **1단계:** 10에포크 동안 높은 컨셉 규제치(`lambda_c = 5.0`)로 학습하고 `phase1_cbm.pth`에 저장합니다.
- **2단계:** 1단계 모델을 로드한 뒤 컨셉 규제치(`lambda_c = 0.5`)를 낮춰 10에포크 더 학습하고 `phase2_cbm.pth`에 저장합니다.

```bash
uv run python main.py --config_path configs/train_config.yaml --epochs 10 --lambda_c 5.0 --save_filename phase1_cbm.pth && \
uv run python main.py --config_path configs/train_config.yaml --epochs 10 --lambda_c 0.5 --resume_checkpoint checkpoints/resnet50/phase1_cbm.pth --save_filename phase2_cbm.pth
```

### 6) CUB-200-2011 다운로드 및 학습
CUB archive를 다운로드하고 검증한 뒤 프로젝트용 concept config를 생성합니다:
```bash
uv run python download_CUB-200-2011.py --data-dir data
uv run python scratch/convert_cub_attributes.py
```

CUB 설정으로 학습합니다. 이 설정에는 LoRA 백본 학습, concept bias calibration, 평가 기본값이 포함됩니다:
```bash
uv run python main.py --config_path configs/cub_train_config.yaml
```

### 7) 컨셉 바이어스 보정
YAML의 `calibration.for_what`에 `learn_concept_bias`를 포함하면 calibration이 활성화됩니다. 학습된 concept-bias buffer는 체크포인트에 저장됩니다:
```yaml
calibration:
  source_split: train
  ratio: 0.10
  seed: 42
  for_what:
    - learn_concept_bias

learn_concept_bias:
  objective:
    metric: target_nll
  parameterization: singleton_only
  temperature: 1.1
  l2_lambda: 0.003
```

### 8) 평가 및 TTI 벤치마크
`eval_cbm.py`로 표준 CBM 평가와 TTI 벤치마크를 실행합니다:
```bash
uv run python eval_cbm.py \
    --checkpoint PATH_TO_CHECKPOINT \
    --config_path configs/cub_train_config.yaml
```

주요 평가 옵션:
```bash
uv run python eval_cbm.py --checkpoint PATH_TO_CHECKPOINT --without-tti
uv run python eval_cbm.py --checkpoint PATH_TO_CHECKPOINT --without-coop-fit --coop-score-mode all
uv run python eval_cbm.py --checkpoint PATH_TO_CHECKPOINT --without-coop-tti
uv run python eval_cbm.py --checkpoint PATH_TO_CHECKPOINT --ignore-bias
```

### 9) Gradio 인터랙티브 앱 기동 (Human-in-the-Loop 검증)
학습된 체크포인트 파일과 클래스 수를 지정해 Gradio 탐색기 웹 인터페이스를 실행합니다:
```bash
uv run python app.py --checkpoint checkpoints/resnet50/phase2_cbm.pth --num_classes 11 --port 7860
```

웹 브라우저로 [http://127.0.0.1:7860](http://127.0.0.1:7860)에 접속하여 다음 기능을 사용할 수 있습니다:
1. 피부 질환 사진을 업로드하고 **예측 확률이 높은 Top-3 질환**을 확인합니다.
2. 범주형 컨셉은 선택된 category 중심으로 정리된 attention map을 확인합니다.
3. 성별/부위 등의 임상 인자는 **dropdown**으로, 나이 등은 **실제 단위 slider**로 수정하여 개입이 예측에 미치는 영향을 확인합니다.

---

## 컨셉 보틀넥 설정 파일 자동 생성

커스텀 데이터셋을 사용할 경우 `generate_concept_config.py`로 메타데이터 CSV를 읽어 JSON 형식의 CBM concept config를 생성할 수 있습니다:

```bash
uv run python generate_concept_config.py \
    --csv_path data/MILK10K/MILK10k_Test_Metadata.csv \
    --ignore_cols lesion_id,image_type,isic_id,attribution,copyright_license,image_manipulation \
    --output_path concept_config.json
```
