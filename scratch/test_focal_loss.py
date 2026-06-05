import torch
import torch.nn.functional as F
from src.utils.losses import SigmoidFocalLoss

def test_sigmoid_focal_loss():
    # 1. Test standard single float alpha
    criterion_float = SigmoidFocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.randn(4, 9)
    targets = torch.randint(0, 2, (4, 9)).float()
    
    loss_float = criterion_float(logits, targets)
    assert isinstance(loss_float, torch.Tensor)
    assert loss_float.ndim == 0
    print("Single float alpha test passed!")

    # 2. Test list of floats alpha
    alpha_list = [0.8, 0.6, 0.95, 0.5, 0.4, 0.7, 0.9, 0.1, 0.3]
    criterion_list = SigmoidFocalLoss(alpha=alpha_list, gamma=2.0)
    assert isinstance(criterion_list.alpha, torch.Tensor)
    assert criterion_list.alpha.shape == (9,)
    
    loss_list = criterion_list(logits, targets)
    assert isinstance(loss_list, torch.Tensor)
    print("List of floats alpha test passed!")

    # 3. Test string of comma-separated floats alpha
    alpha_str = "0.8, 0.6, 0.95, 0.5, 0.4, 0.7, 0.9, 0.1, 0.3"
    criterion_str = SigmoidFocalLoss(alpha=alpha_str, gamma=2.0)
    assert isinstance(criterion_str.alpha, torch.Tensor)
    assert torch.allclose(criterion_str.alpha, torch.tensor(alpha_list))
    
    loss_str = criterion_str(logits, targets)
    assert torch.allclose(loss_list, loss_str)
    print("String of floats alpha test passed!")

    # 4. Check exact math formula
    # Let's manually compute for a simple batch size = 2, concept size = 2
    alpha_test = [0.8, 0.3]
    criterion_test = SigmoidFocalLoss(alpha=alpha_test, gamma=2.0)
    
    test_logits = torch.tensor([[0.5, -0.5], [1.0, -1.0]], dtype=torch.float32)
    test_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    
    # Run through the loss
    loss_val = criterion_test(test_logits, test_targets)
    
    # Manual calculation
    # bce_loss
    bce = F.binary_cross_entropy_with_logits(test_logits, test_targets, reduction='none')
    # p_t
    p_t = torch.exp(-bce)
    # alpha_t
    alpha_tensor = torch.tensor(alpha_test, dtype=torch.float32)
    alpha_t = alpha_tensor * test_targets + (1.0 - alpha_tensor) * (1.0 - test_targets)
    # focal loss per element
    expected_focal = alpha_t * ((1.0 - p_t) ** 2.0) * bce
    expected_loss = expected_focal.mean()
    
    print(f"Calculated loss: {loss_val.item():.6f}, Expected loss: {expected_loss.item():.6f}")
    assert torch.allclose(loss_val, expected_loss)
    print("Exact mathematical formula test passed!")

if __name__ == "__main__":
    test_sigmoid_focal_loss()
    print("ALL FOCAL LOSS TESTS PASSED!")
