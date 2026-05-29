# Request: Create a Production-Grade PyTorch CBM DataLoader for [데이터셋 이름]

안녕하세요! 저는 [데이터셋 이름] 데이터셋을 사용하여 어텐션 기반 컨셉 보틀넥 모델(Attention-CBM)을 학습하려고 합니다.
다음 상세 가이드라인과 설계 규칙을 100% 충족하는 고도의 모듈형 PyTorch Dataset 클래스를 작성해 주세요.

## 1. 데이터셋 명세
- **데이터셋 이름**: [예: ISIC 2019 / HAM10000 등]
- **메타데이터 CSV 경로**: [예: data/ISIC2019/ISIC_2019_Training_Metadata.csv]
- **그라운드 트루스 CSV 경로**: [예: data/ISIC2019/ISIC_2019_Training_GroundTruth.csv (라벨 파일이 분리되어 있다면 입력, 없으면 null)]
- **이미지 폴더 구조**: [예: data/images 하위에 lesion_id별 폴더가 있고 그 안에 ISIC_id.jpg 파일이 있는 중첩 구조]
- **컨셉 컬럼 명세**: [예: MONET_ 으로 시작하는 수치형 컬럼들 및 age_approx, sex 등의 수치형/범주형 속성]
- **타겟 컬럼 명세**: [예: Malignancy (MEL, BCC, AKIEC 등 악성 암 종을 1.0으로 병합한 바이너리 라벨)]

---

## 2. 필수 구현 및 설계 규칙 (Strict Rules)

### ① 데이터셋 분할 및 누수 차단 (Leak-Proof Splitting)
- Train(70%), Val(15%), Test(15%) 분할은 일관되게 `random_state=42`로 셔플한 뒤 인덱스를 슬라이싱하여 **서로 완벽히 격리(Disjoint)**되도록 구현해 주세요.
- 만약 폴더 내에 `train_indexes.csv` 등 공식 분할 파일이 감지된다면 우선적으로 이를 참조하여 인덱스를 수집해 주세요.

### ② 메타데이터와 정답 라벨 파일의 자동 병합
- 메타데이터 CSV와 그라운드 트루스 라벨 CSV가 별도 파일로 분리되어 제공된 경우, Dataset 초기화 시 `lesion_id` 혹은 지정된 키를 기반으로 자동 병합(`pd.merge`) 하도록 코드를 작성해 주세요.
- 다중 클래스(Multi-class) 정답 라벨이 주어졌을 때 바이너리 타겟(Malignancy 등) 컬럼을 지정된 악성 컬럼들의 합이나 합집합 조건으로 자동 가공하는 로직을 생성자 내부에 추가해 주세요.

### ③ 시각화용 1차원 컨셉 리스트(`concepts_flat`) 구성
- **중요**: 범주형 컨셉(예: `sex` $\rightarrow$ `male, female`)이 원-핫 인코딩되어 실제 모델의 보틀넥(예: 22차원)과 원본 컨셉 개수(예: 11개)가 달라지면 검증 단계 히트맵 시각화 시 `IndexError`가 발생합니다.
- 원-핫 인코딩의 차원 확장을 고려하여, 실제 보틀넥의 각 노드에 매핑될 평탄화된 컨셉 컬럼명 리스트 `self.config["concepts_flat"]` (예: `['MONET_erythema', 'sex_male', 'sex_female']` 등 22개 차원)을 실시간으로 빌드하여 반환하도록 설계해 주세요.

### ④ 다중 이미지 경로 추적 및 안정성 보장 (Graceful Exception Handling)
- 이미지는 `image_dir/lesion_id/isic_id.jpg` (중첩 구조)와 `image_dir/isic_id.jpg` (플랫 구조) 경로를 모두 동적으로 체크(`os.path.exists`)하여 로드할 수 있게 만드세요.
- 루프 중 간혹 발생하는 이미지 유실이나 경로 손상 시 전체 프로세스가 죽지 않도록 `FileNotFoundError` 등을 예외 처리하고, 유실 경고를 딱 한 번만 출력한 뒤 `(224, 224)` 규격의 검은색 더미 이미지를 생성해 반환하도록 안전 로직을 구현해 주세요.

### ⑤ 인메모리 캐싱 (In-Memory Caching)
- **중요**: `BaseDataset`은 인메모리 캐싱을 지원합니다. 서브클래스는 반드시 다음 패턴을 따라야 합니다:
  - `__getitem__`이 아닌 **`_load_sample(self, idx)`** 메서드에 실제 데이터 로딩 로직(이미지 읽기, 컨셉 텐서 빌드, 타겟 텐서 빌드)을 구현해 주세요.
  - `__getitem__`은 `BaseDataset`에서 이미 구현되어 있으며, 캐시가 있으면 캐시에서 반환하고 없으면 `_load_sample`을 호출합니다. **서브클래스에서 `__getitem__`을 오버라이드하지 마세요.**
  - `__init__` 시그니처에 `cache_in_memory: bool = False`와 `max_cache_size_gb: float = 10.0` 파라미터를 추가해 주세요.
  - `__init__`의 **맨 마지막**에 다음 세 줄을 반드시 추가해 주세요:
    ```python
    self.cache_in_memory = cache_in_memory
    self._cache = None
    self._cache_populated = False
    self._try_populate_cache(max_cache_size_gb=max_cache_size_gb)
    ```
  - 캐싱 동작 원리:
    1. `_try_populate_cache()`가 `self.image_dir`의 디스크 용량을 자동 추정합니다.
    2. 데이터셋이 `max_cache_size_gb` (기본 10GB) 미만이면 모든 샘플을 tqdm 프로그레스 바와 함께 RAM에 프리로드합니다.
    3. 학습 중 `__getitem__`은 디스크 I/O 없이 캐시에서 즉시 반환합니다.
    4. 데이터셋이 임계값보다 크면 자동으로 on-the-fly 로딩으로 폴백합니다.

---

## 3. 코드 구조 템플릿
다음 형태의 `BaseDataset`을 상속받는 완성된 Python 코드로 제공해 주세요:

```python
import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from src.data.base_dataset import BaseDataset

class CustomCBMDataset(BaseDataset):
    @classmethod
    def get_default_config(cls) -> dict:
        # 기본 설정을 딕셔너리로 반환
        pass

    def __init__(
        self,
        csv_path=None,
        image_dir=None,
        split='train',
        config=None,
        transform=None,
        cache_in_memory: bool = False,
        max_cache_size_gb: float = 10.0
    ):
        # 1. 자동 병합 및 인덱스 분할
        # 2. concepts_flat 빌드
        # 3. 데이터 가공 및 정규화
        # 4. 인메모리 캐싱 초기화 (맨 마지막에 호출!)
        self.cache_in_memory = cache_in_memory
        self._cache = None
        self._cache_populated = False
        self._try_populate_cache(max_cache_size_gb=max_cache_size_gb)

    def __len__(self):
        # 데이터셋 길이 반환 (더미 모드 지원 포함)
        pass

    # ⚠️ __getitem__은 BaseDataset에서 캐시-어웨어로 구현되어 있으므로
    #    서브클래스에서 오버라이드하지 않습니다!

    def _load_sample(self, idx):
        # 1. 동적 경로 추적 및 안전한 이미지 오픈
        # 2. 컨셉 텐서 빌드 (concepts_flat 기반)
        # 3. 타겟 텐서 빌드
        # return image, concept_tensor, target_tensor
        pass
```

## 4. BaseDataset API 참고

`BaseDataset`(`src/data/base_dataset.py`)은 다음 API를 제공합니다:

| 메서드/속성 | 타입 | 설명 |
|---|---|---|
| `get_default_config()` | `@classmethod` | 기본 config dict 반환 (서브클래스 오버라이드 필수) |
| `_load_sample(idx)` | `@abstractmethod` | 디스크에서 단일 샘플 로드 (서브클래스 구현 필수) |
| `__len__()` | `@abstractmethod` | 데이터셋 길이 반환 (서브클래스 구현 필수) |
| `__getitem__(idx)` | 구현됨 | 캐시 히트 시 캐시 반환, 미스 시 `_load_sample` 호출 |
| `_estimate_dataset_size_gb()` | 구현됨 | `self.image_dir`의 디스크 용량(GB) 추정 |
| `_try_populate_cache(max_cache_size_gb)` | 구현됨 | 캐시 조건 확인 후 전체 샘플 프리로드 |
| `self.cache_in_memory` | `bool` | 캐싱 활성화 여부 (서브클래스에서 설정) |
| `self._cache` | `list \| None` | 캐시 저장소 |
| `self._cache_populated` | `bool` | 캐시 완료 여부 플래그 |
