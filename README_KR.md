# 어텐션 기반 모듈형 컨셉 보틀넥 모델 (Attention-CBM) 파이프라인

본 프로젝트는 의료 영상 분석 및 미세 시각 분류(Fine-grained visual categorization)를 위해 설계된 고도의 모듈형 **어텐션 기반 컨셉 보틀넥 모델(Attention-CBM)** 파이프라인입니다.

---

## 1. 주요 특징 및 모델 구조

기존의 전역 평균 풀링(Global Average Pooling, GAP) 방식을 버리고, 이미지의 특정 공간 영역을 각 컨셉이 학습하여 바라볼 수 있도록 **교차 어텐션(Cross-Attention) 메커니즘**을 도입하였습니다. 이를 통해 픽셀 수준의 설명 가능성(Heatmap 시각화)을 보장합니다.

### 💡 공간적 컨셉 그라운딩 (Spatial Concept Grounding)
- **ConceptAttentionLayer:** `nn.MultiheadAttention`을 기반으로 설계되었습니다.
- **Concept Queries:** 학습 가능한 파라미터 `[1, num_concepts, feature_dim]`가 어텐션의 쿼리(Query)로 작동합니다.
- **Spatial Keys & Values:** GAP를 거치지 않은 백본의 2D 공간 특징 맵 `[B, C, H, W]`을 `[B, H*W, C]` 형상으로 변환하여 키(Key) 및 값(Value)으로 입력합니다.
- **Explainability:** 컨셉별 어텐션 가중치 `[B, num_concepts, H, W]`가 함께 반환되므로, 특정 컨셉을 예측할 때 모델이 이미지의 어떤 영역(Patch)을 집중했는지 히트맵으로 시각화할 수 있습니다.

### ⚙️ 유연한 백본 지원 (timm)
- `timm` 라이브러리를 통해 다양한 최신 백본 네트워크를 지원합니다.
- 어텐션 적용을 위해 `global_pool=''` 옵션을 통해 전역 풀링을 바이패스하고 원본 2D 피처 맵을 그대로 추출합니다.
- 입력 이미지 해상도에 맞춰 특징 차원 `C`와 공간 차원 `(H, W)`을 동적으로 추론하므로 별도의 하드코딩이 필요 없습니다.
- *(참고: open_clip은 2D 공간 특징 맵 추출 방식의 일관성 문제로 현재 Attention-CBM에서는 지원되지 않으며, timm 백본 사용을 권장합니다.)*

### 📉 하이브리드 손실 함수 (Hybrid Loss Training)
- 분류 대상 타겟 손실 함수(Target Classification Loss)와 보틀넥 컨셉 손실 함수(Concept Loss)를 결합한 하이브리드 손실을 학습합니다.
- `--lambda_c` 가중치 인자를 통해 컨셉 학습의 강도를 손쉽게 조절할 수 있습니다.

---

## 2. 실행 방법 (Usage Examples)

`main.py` 진입점을 활용하여 학습 파이프라인을 실행합니다. 제공된 컨셉 설정 파일(`--concept_config_path`)의 데이터 규격에 맞추어 보틀넥 레이어가 자동으로 설정됩니다.

### 1) timm 백본을 활용한 기본 학습 (ResNet50)
```bash
uv run python main.py \
    --dataset milk10k \
    --concept_config_path data/MILK10K/concept_config.json \
    --backbone_type timm \
    --backbone_name resnet50 \
    --lambda_c 1.0 \
    --num_classes 1 \
    --epochs 5 \
    --batch_size 16
```

### 2) 순차 학습 (비전 백본 가중치 동결)
비전 백본을 동결(Freeze)하고 어텐션 쿼리, 프로젝션 레이어 및 최종 분류 헤드만 학습시키는 모드입니다.
```bash
uv run python main.py \
    --dataset milk10k \
    --concept_config_path data/MILK10K/concept_config.json \
    --backbone_type timm \
    --backbone_name convnext_base \
    --freeze_backbone
```

### 3) Weights & Biases (wandb) 로깅 및 백본 구분
기본적으로 wandb 로깅이 활성화되어 있으며, 각 실험(run)의 이름은 다음과 같이 백본명과 실행 타임스탬프를 조합하여 자동으로 가독성 있게 생성됩니다.
- 예: `resnet50-cbm-20260530_010616`

또한, `main.py`에 넘겨진 모든 CLI 인자(`backbone_name`, `backbone_type`, `lambda_c` 등)가 wandb의 `config`에 자동으로 로깅되므로, W&B 대시보드 내에서 각 실험이 어떤 백본을 사용했는지 필터링, 그룹화 및 확인이 가능합니다.

wandb 로깅을 비활성화하려면 `--use_wandb False` 옵션을 전달합니다.
```bash
uv run python main.py \
    --dataset milk10k \
    --concept_config_path data/MILK10K/concept_config.json \
    --backbone_type timm \
    --backbone_name resnet50 \
    --use_wandb False
```

---

## 3. 모델 가중치 저장 (Model Weight Saving)

학습이 완료되면, 모델의 가중치(`state_dict`)가 지정한 `--save_dir` 디렉토리 아래에 **백본 이름별 서브디렉토리**로 구분되어 자동으로 저장됩니다. 파일명에는 실행 시점의 타임스탬프와 백본 가중치 동결 여부(mode)가 포함되어 구분됩니다.

- **기본 저장 디렉토리:** `checkpoints/{backbone_name}/`
- **저장 파일명 규칙:** `{YYYYMMDD_HHMMSS}_cbm_{mode}.pth` (예: `20260530_010616_cbm_full.pth` 또는 `20260530_011045_cbm_frozen_backbone.pth`)
- **CLI 옵션으로 저장 경로 변경:** `--save_dir [경로]` (기본값: `checkpoints`)

---

## 3. 컨셉 설정 파일 자동 생성 (Auto-Generating Concept Config)

제공된 `generate_concept_config.py` 스크립트를 사용하여 데이터셋의 메타데이터 CSV 파일 분석 결과를 기반으로 어텐션 보틀넥에 전달할 JSON 형식의 설정 파일을 자동으로 생성할 수 있습니다.

```bash
uv run python generate_concept_config.py \
    --csv_path data/MILK10K/MILK10k_Test_Metadata.csv \
    --ignore_cols lesion_id,image_type,isic_id,attribution,copyright_license,image_manipulation \
    --output_path concept_config.json
```

> [!TIP]
> **수동 수정 가이드:** 스크립트로 자동 생성된 JSON/YAML 파일은 훌륭한 초기 기준점을 제공합니다. 생성 후 범주형(Categorical)과 수치형(Numerical) 타입을 바꾸거나, min-max 경계 조정, 불필요한 컨셉 가지치기 등 자유롭게 직접 열어 수동 조정하여 사용할 수 있습니다.
