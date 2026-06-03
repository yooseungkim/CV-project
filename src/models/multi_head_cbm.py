import torch
import torch.nn as nn
from typing import Dict, Any

class MultiHeadConceptExtractor(nn.Module):
    """
    MultiHeadConceptExtractor: 설정 파일(concept_config) 기반 다중 헤드 컨셉 추출기.
    각 카테고리별로 상호 배타적인(Softmax) 특징 추출을 위해 독립적인 분류 헤드를 구성합니다.
    """
    def __init__(self, in_features: int, concept_config: Dict[str, int]):
        super().__init__()
        self.concept_config = concept_config
        self.in_features = in_features
        
        # 각 카테고리별 리니어 헤드를 저장할 ModuleDict 생성
        self.heads = nn.ModuleDict()
        
        for name, num_classes in concept_config.items():
            if num_classes < 1:
                raise ValueError(f"Invalid class count for {name}: {num_classes}")
            # nn.ModuleDict 내에서 안전하게 이름을 사용하기 위해 리니어 레이어 매핑
            self.heads[name] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: Backbone으로부터 나온 피처 맵 혹은 풀링된 텐서 [BatchSize, in_features]
        반환값: 카테고리명을 키로 하고 예측 로그잇을 값으로 하는 딕셔너리
        """
        # Loss 함수에서 Softmax를 계산할 수 있도록 원시 Logits 반환
        return {name: head(x) for name, head in self.heads.items()}


class CategoricalConceptLoss(nn.Module):
    """
    CategoricalConceptLoss: 각 카테고리별 CrossEntropyLoss 및 회귀 MSELoss를 집계하는 커스텀 손실 함수.
    """
    def __init__(self, aggregation: str = 'sum'):
        super().__init__()
        if aggregation not in ['sum', 'mean']:
            raise ValueError("aggregation must be 'sum' or 'mean'")
        self.aggregation = aggregation
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_logits: Dict[str, torch.Tensor], target_concepts: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        pred_logits: MultiHeadConceptExtractor의 결과물인 {카테고리명: 예측 로그잇 텐서}
        target_concepts: Ground Truth 레이블 {카테고리명: 타겟 텐서}
        """
        total_loss = torch.tensor(0.0, device=next(iter(pred_logits.values())).device)
        count = 0
        
        for name, pred in pred_logits.items():
            if name not in target_concepts:
                continue
                
            target = target_concepts[name]
            num_classes = pred.shape[-1]
            
            if num_classes == 1:
                # 회귀 헤드: MSELoss 계산
                # 차원 일치를 위해 예측 및 타겟을 1차원 텐서로 통일
                loss = self.mse_loss(pred.squeeze(-1), target.float())
            else:
                # 다중 분류 헤드: CrossEntropyLoss 계산
                loss = self.ce_loss(pred, target.long())
                
            total_loss = total_loss + loss
            count += 1
            
        if count == 0:
            return total_loss
            
        if self.aggregation == 'mean':
            total_loss = total_loss / count
            
        return total_loss


# ==============================================================================
# 단위 테스트 코드 (Structural Unit Test)
# ==============================================================================
if __name__ == "__main__":
    print("Executing structural unit test for Multi-Head CBM Extractor...")
    
    # 1. 테스트 설정 구성 (다중 분류 카테고리 + 회귀 카테고리 혼합)
    dummy_concept_config = {
        'bill_shape': 9,         # Categorical (9 classes)
        'wing_color': 15,        # Categorical (15 classes)
        'is_spotted': 1,         # Regression / Continuous Scalar
        'ulceration_type': 3     # Categorical (3 classes)
    }
    
    in_features = 512
    batch_size = 8
    
    # 2. 모듈 인스턴스화
    extractor = MultiHeadConceptExtractor(in_features=in_features, concept_config=dummy_concept_config)
    criterion = CategoricalConceptLoss(aggregation='mean')
    
    # 3. 더미 입력 데이터 생성 [BatchSize, in_features]
    dummy_backbone_features = torch.randn(batch_size, in_features, requires_grad=True)
    
    # 4. 더미 Ground Truth 레이블 생성
    dummy_targets = {
        'bill_shape': torch.randint(0, 9, (batch_size,)),
        'wing_color': torch.randint(0, 15, (batch_size,)),
        'is_spotted': torch.rand(batch_size), # 회귀용 실수 값
        'ulceration_type': torch.randint(0, 3, (batch_size,))
    }
    
    # 5. 순전파 수행 (Forward Pass)
    pred_logits = extractor(dummy_backbone_features)
    
    # 6. 예측값 차원 검증
    print("\n--- Shape Verification ---")
    for name, logits in pred_logits.items():
        print(f"Category: {name:<17} | Output Shape: {str(list(logits.shape)):<12} | Target Shape: {str(list(dummy_targets[name].shape))}")
        
        # 차원 검증 단언문
        expected_classes = dummy_concept_config[name]
        assert logits.shape == (batch_size, expected_classes), f"Shape mismatch for {name}"
        
    # 7. 손실값 계산 및 역전파 검증 (Loss & Backward Pass)
    loss = criterion(pred_logits, dummy_targets)
    print(f"\nCalculated Aggregated Loss: {loss.item():.6f}")
    
    loss.backward()
    
    # 입력 텐서의 그래디언트 유입 상태 검증
    assert dummy_backbone_features.grad is not None, "Gradient did not flow back to input features!"
    print("Gradient successfully backpropagated to input features.")
    
    # 파라미터 그래디언트 상태 검증
    for name, head in extractor.heads.items():
        assert head.weight.grad is not None, f"Gradient not computed for head: {name}"
    print("Gradients computed successfully for all heads parameters.")
    
    print("\nUnit Test successfully passed!")
