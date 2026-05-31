import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))

from src.models.cbm_factory import UniversalFlexibleCBM

def test_latent_losses():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running test on device: {device}")
    
    # 1. Instantiate model with 5 latent concepts and 112 supervised concepts
    print("Initializing CBM model with DINOv2 backbone and LoRA...")
    model = UniversalFlexibleCBM(
        backbone_type='timm',
        backbone_name='vit_base_patch14_dinov2',
        num_supervised_concepts=112,
        num_classes=200,
        num_latent_concepts=5,
        use_lora=True
    )
    model.to(device)
    model.eval()
    
    # Simulate a dummy image batch: [B, 3, 224, 224]
    x = torch.randn(2, 3, 224, 224, device=device)
    
    print("\n1. Standard forward pass compatibility check (expecting 3 return values):")
    class_logits, concept_probs, attn_weights = model(x)
    print("Class logits shape:", class_logits.shape)
    print("Concept probs shape:", concept_probs.shape)
    print("Attention weights shape:", attn_weights.shape)
    assert class_logits.shape == (2, 200)
    assert concept_probs.shape == (2, 117)
    assert attn_weights.shape == (2, 117, 16, 16)
    print("🎉 Standard forward pass is fully backward compatible!")

    print("\n2. Optional forward pass check with return_features=True:")
    class_logits, concept_probs, attn_weights, explicit_features, latent_features = model(x, return_features=True)
    print("Explicit pre-activation features shape:", explicit_features.shape)
    print("Latent pre-activation features shape:", latent_features.shape)
    assert explicit_features.shape == (2, 112, 768)
    assert latent_features.shape == (2, 5, 768)
    print("🎉 pre-activation features shape validation PASSED!")

    print("\n3. Phase 2 Simulation & Latent Loss gradient flow check:")
    # Freeze explicit concepts & backbone parameters
    model.freeze_backbone()
    model.freeze_supervised_attention()
    model.unfreeze_latent_attention()
    
    # Check what parameters require gradient
    grad_params = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters in Phase 2 (sample count: {len(grad_params)}):")
    for name in grad_params[:3]:
        print(f" - {name}")
    assert any("latent_attention" in name for name in grad_params), "Latent attention should be trainable!"
    assert not any("supervised_attention" in name for name in grad_params), "Supervised attention should be frozen!"
    
    # Phase 2 step simulation
    features = model.backbone(x)
    with torch.no_grad():
        supervised_logits, _, supervised_features = model.supervised_attention(features)
        
    # Unfrozen path
    latent_logits, _, latent_features = model.latent_attention(features)
    concept_logits = torch.cat([supervised_logits, latent_logits], dim=1)
    concept_probs = model.concept_activation(concept_logits)
    
    # Calculate Orthogonal Projection Loss
    explicit_norm = F.normalize(supervised_features, p=2, dim=-1)  # [B, S, C]
    latent_norm = F.normalize(latent_features, p=2, dim=-1)        # [B, L, C]
    
    cos_sim = torch.bmm(latent_norm, explicit_norm.transpose(1, 2))  # [B, L, S]
    loss_latent_ortho = (cos_sim ** 2).mean()
    print("Computed Orthogonal latent loss:", loss_latent_ortho.item())
    
    # Calculate Latent Sparsity Loss (L1)
    latent_activations = concept_probs[:, 112:]  # [B, L]
    loss_latent_l1 = latent_activations.abs().mean()
    print("Computed L1 latent sparsity loss:", loss_latent_l1.item())
    
    # Total loss backprop
    total_loss = 0.1 * loss_latent_ortho + 0.01 * loss_latent_l1
    print("Simulated total latent loss:", total_loss.item())
    
    total_loss.backward()
    
    # Assert that gradients flowed to latent parameters and are non-zero!
    latent_grad_sum = sum(p.grad.abs().sum().item() for p in model.latent_attention.parameters() if p.grad is not None)
    print("Latent attention gradient magnitude sum:", latent_grad_sum)
    assert latent_grad_sum > 0.0, "Gradients must flow to latent query layers!"
    
    # Assert that explicit query parameters have NO gradients!
    for name, p in model.supervised_attention.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().sum().item() == 0.0, f"Supervised attention parameter {name} should have zero gradient!"
            
    print("🎉 ALL LATENT DISENTANGLEMENT & GRADIENT ASSERTIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_latent_losses()
