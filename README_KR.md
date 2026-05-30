# 어텐션 기반 모듈형 컨셉 보틀넥 모델 (Attention-CBM) 파이프라인

본 프로젝트는 의료 영상 분석 및 미세 시각 분류(Fine-grained visual categorization)를 위해 설계된 고도의 모듈형 **어텐션 기반 컨셉 보틀넥 모델(Attention-CBM)** 파이프라인입니다.

---

## 🚀 주요 프리미엄 핵심 기능

- **공간적 컨셉 그라운딩 (Spatial Concept Grounding):** 전역 평균 풀링(GAP) 방식을 버리고, 이미지의 2D 공간 특징 맵 위에 학습 가능한 컨셉 쿼리를 교차 어텐션(Cross-Attention, `nn.MultiheadAttention`)하여 적용함으로써 픽셀 수준의 어텐션 설명 가능성(히트맵)을 완벽히 제공합니다.
- **다중 질환 분류 파이프라인 (Multi-Class Disease Classification):** 단순 이진 분류(Benign/Malignant)를 넘어 MILK10K의 GroundTruth CSV를 연동하여 11개 핵심 피부 질환 카테고리(`AKIEC`, `BCC`, `BEN_OTH`, `BKL`, `DF`, `INF`, `MAL_OTH`, `MEL`, `NV`, `SCCKA`, `VASC`) 다중 분류 체계로 전면 전환 및 고도화했습니다.
- **Inverse-Frequency 클래스 불균형 완화:** 극심한 클래스 불균형(예: BCC 2522개 vs. MAL_OTH 9개) 문제를 해결하기 위해, 학습 데이터셋의 분포를 추적해 역빈도 클래스 가중치(Inverse-Frequency Weights)를 계산하고 이를 `nn.CrossEntropyLoss`에 연동했습니다.
- **모델 정규화 (Dropout):** 컨셉 보틀넥의 예측 활성화 벡터에 `nn.Dropout(p=0.2)` 레이어를 결합하여 가중치 로드의 완벽한 하위 호환성을 유지하면서 노이즈 데이터를 배제하고 타겟 분류 강건성을 대폭 향상했습니다.
- **Gradio 웹 애플리케이션 및 Human-in-the-Loop 혁신:**
  - **필터링된 컨셉 어텐션 맵 시각화:** 동일 범주에 속하는 컨셉(예: `site_foot`, `site_trunk` 등은 `site` 컨셉 그룹에 대응) 중에서 **가장 높은 예측 확률(argmax)을 획득한 1개의 클래스만 시각화**하여 UI의 복잡함을 없앴습니다 (22개 열 대신 딱 11개 주요 컨셉만 표시).
  - **범주형 드롭다운 컴포넌트:** 여러 개의 무의미한 $[0.0, 1.0]$ 개별 슬라이더를 **통합 드롭다운(Dropdown)**으로 묶어 pre-select하고, 사용자가 값을 바꿀 때 백엔드에서 원-핫 인코딩으로 자동 변환해 줍니다.
  - **실제 물리 스케일 슬라이더:** 수치형 컨셉(예: 나이)은 `concept_config.json` 내의 실제 연령 범위($5$ ~ $85$세) 및 MONET 컨셉 실수 스케일을 추적해 구현되었습니다.
    * *순방향 예측:* 모델의 $[0, 1]$ 시그모이드 출력을 실제 연령/스케일 값으로 복원하여 입력창에 표시합니다.
    * *역방향 개입:* 사용자가 조정한 실제 값을 다시 $[0, 1]$ 범위로 정규화(Min-Max)해 모델 헤드에 인코딩 주입합니다.
  - **초압축 2열(2-column) 레이아웃:** 콤팩트하게 다듬은 CSS와 좌우 2열 배치 그리드를 설계하여, 스크롤 없이 전체 조절 박스가 한눈에 들어오도록 고급스럽게 디자인했습니다.
- **강력한 전이 학습 및 모델 재개 (Resume Checkpoint):**
  - CLI 및 YAML 설정 파일에 `--resume_checkpoint` 옵션을 추가했습니다.
  - **하이브리드 가중치 로드 전략**을 적용하여 일치 시 `strict=True`로 로드하고, 분류 헤드 변경 등 형상이 달라지면 `strict=False`로 전환해 백본과 컨셉 어텐션 레이어만 전이(Fine-tuning)될 수 있도록 처리했습니다.
- **유연한 가중치 파일명 지정:** `--save_filename` 인자를 지원하여 다단계 연쇄 학습 파이프라인 스크립팅을 자동화했습니다.
- **상세 컨셉 학습 모니터링:**
  - 검증 에포크가 끝날 때마다 **22개 개별 컨셉의 고유 검증 정확도**를 W&B에 개별로 전송합니다 (`val_concept_acc/{concept_name}`).
  - 터미널 창에 현재 모델이 가장 학습을 어려워하는 **Struggling Concepts Top-3**를 실시간 분석해 실시간 출력합니다.
- **초고속 RAM 메모리 캐싱:** `cache_in_memory: true` 지원으로 대용량 학습 시 디스크 I/O 병목을 완벽하게 제거했습니다.

---

## 디렉토리 구조
```
project_root/
├── checkpoints/            # 학습된 백본 모델별 가중치 저장소
├── configs/
│   └── train_config.yaml   # 에포크, LR, 옵티마이저, 스케줄러, 얼리스톱 통합 설정 YAML
├── data/                   # (git 제외) MILK10K 로우 데이터 저장소
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── base_dataset.py # 추상 데이터셋 기본 클래스
│   │   └── milk10k.py      # MILK10K 다중 클래스 데이터셋 로더
│   ├── models/
│   │   ├── __init__.py
│   │   └── cbm_factory.py  # UniversalFlexibleCBM 레이아웃 빌더
│   └── utils/
│       ├── __init__.py
│       └── metrics.py      # 정확도 및 평가지표 연산 유틸
├── app.py                  # Gradio 인터랙티브 웹 익스플로러 어플리케이션
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

`main.py` 진입점을 활용하여 학습 파이프라인을 실행합니다. 학습 시 `--config_path`에 정의된 YAML 설정값을 기반으로 레이아웃과 파라미터가 최적화되어 자동 설정됩니다.

### 1) 기본 CBM 학습 (설정 파일 값 그대로 사용)
비전 백본, 컨셉 어텐션 쿼리, 최종 분류기 헤드를 `configs/train_config.yaml`에 정의된 기본 최적 파라미터 규격으로 일괄 학습합니다:
```bash
uv run python main.py --config_path configs/train_config.yaml
```

### 2) 기본 CBM 학습과 CLI 파라미터 덮어쓰기 (Overrides)
기본 설정을 기반으로 하되, 특정 파라미터(에포크, 배치 사이즈, 컨셉 가중치, 메인 헤드 학습률 등)만 명령행에서 동적으로 조절하여 기동합니다:
```bash
uv run python main.py \
    --config_path configs/train_config.yaml \
    --epochs 15 \
    --batch_size 32 \
    --lambda_c 3.0 \
    --lr 0.0005
```

### 3) 순차 학습: 비전 백본 동결 (Backbone Freeze)
이미 검증된 비전 백본 특징 추출기 가중치를 완전 동결하고, **오직** 교차 공간 어텐션 레이어와 분류 헤드만을 학습시킵니다 (백본 손상 방지 및 학습 속도 대폭 개선):
```bash
uv run python main.py --config_path configs/train_config.yaml --freeze_backbone
```

### 4) 순차 학습: 분류기 헤드 동결 (Classifier Head Freeze)
최종 분류기 헤드를 고정시키고 **오직** 비전 백본 미세조정(Fine-tuning) 및 어텐션 컨셉 그라운딩 레이어만 업데이트합니다:
```bash
uv run python main.py --config_path configs/train_config.yaml --freeze_head
```

### 5) 2단계 연쇄 연속 학습 (Sequential Schedule Training)
`--save_filename` 및 `--resume_checkpoint` 옵션을 사용하여 여러 단계의 학습을 연쇄적으로 결합하여 기동합니다.
- **1단계:** 10에포크 동안 높은 컨셉 규제치(`lambda_c = 5.0`)로 강하게 컨셉 보틀넥 특징을 강제 주입하여 `phase1_cbm.pth`에 저장합니다.
- **2단계:** 1단계 모델을 이어서 로드한 후, 규제를 대폭 완화하여(`lambda_c = 0.5`) 추가 10에포크 동안 최종 예측 정확도를 끌어올리고 `phase2_cbm.pth`로 완성합니다.

```bash
uv run python main.py --config_path configs/train_config.yaml --epochs 10 --lambda_c 5.0 --save_filename phase1_cbm.pth && \
uv run python main.py --config_path configs/train_config.yaml --epochs 10 --lambda_c 0.5 --resume_checkpoint checkpoints/resnet50/phase1_cbm.pth --save_filename phase2_cbm.pth
```

### 6) Gradio 인터랙티브 앱 기동 (Human-in-the-Loop 검증)
학습이 끝난 최종 체크포인트 파일을 지정하고 클래스 규격을 맞춰 Gradio 탐색기 웹 인터페이스를 기동합니다:
```bash
uv run python app.py --checkpoint checkpoints/resnet50/phase2_cbm.pth --num_classes 11 --port 7860
```

웹 브라우저로 [http://127.0.0.1:7860](http://127.0.0.1:7860)에 접속하여 다음 기능을 누릴 수 있습니다:
1. 피부 질환 사진을 드래그하여 업로드하면 **예측 확률이 높은 Top-3 질환**을 즉시 표시합니다.
2. 불필요한 노이즈가 제거되고 **예측된 최적 범주에 대한 컨셉 어텐션 시각화 히트맵**만 직관적으로 확인합니다.
3. 성별/부위 등의 임상 인자는 **드롭다운**으로, 나이 등은 **실물 단위 슬라이더**로 손쉽게 값을 수동 개입하여 예측이 어떻게 달라지는지 웹 UI에서 실시간 실험해 봅니다.

---

## 컨셉 보틀넥 설정 파일 자동 생성

커스텀 데이터셋을 활용할 경우, `generate_concept_config.py` 스크립트를 돌려 데이터의 메타데이터 CSV를 완벽히 자동 스캔하여 JSON 규격의 CBM 가이드 파일을 얻을 수 있습니다:

```bash
uv run python generate_concept_config.py \
    --csv_path data/MILK10K/MILK10k_Test_Metadata.csv \
    --ignore_cols lesion_id,image_type,isic_id,attribution,copyright_license,image_manipulation \
    --output_path concept_config.json
```
