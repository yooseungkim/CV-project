import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))

from src.models.cbm_factory import UniversalFlexibleCBM

def test_latent_losses():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running CBM training pipeline tests on device: {device}")
    
    # 1. Instantiate model with 5 latent concepts and 112 supervised concepts
    print("\nInitializing CBM model with DINOv2 backbone and LoRA...")
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
    dummy_concepts = torch.randint(0, 2, (2, 112), dtype=torch.float32, device=device)
    dummy_targets = torch.randint(0, 200, (2,), dtype=torch.long, device=device)
    
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
    model.unfreeze_classifier() # Classifier is unconstrained in Phase 2
    
    # Check what parameters require gradient
    grad_params_p2 = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters in Phase 2 (sample count: {len(grad_params_p2)}):")
    for name in grad_params_p2[:3]:
        print(f" - {name}")
    assert any("latent_attention" in name for name in grad_params_p2), "Latent attention should be trainable in Phase 2!"
    assert not any("supervised_attention" in name for name in grad_params_p2), "Supervised attention should be frozen in Phase 2!"
    assert not any("backbone" in name for name in grad_params_p2), "Backbone should be frozen in Phase 2!"
    
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
    
    # Calculate Latent Sparsity Loss (L1)
    latent_activations = concept_probs[:, 112:]  # [B, L]
    loss_latent_l1 = latent_activations.abs().mean()
    
    # Total loss backprop
    total_loss_p2 = 0.1 * loss_latent_ortho + 0.01 * loss_latent_l1
    model.zero_grad()
    total_loss_p2.backward()
    
    # Assert that gradients flowed to latent parameters and are non-zero!
    latent_grad_sum = sum(p.grad.abs().sum().item() for p in model.latent_attention.parameters() if p.grad is not None)
    print("Latent attention gradient magnitude sum (Phase 2):", latent_grad_sum)
    assert latent_grad_sum > 0.0, "Gradients must flow to latent query layers in Phase 2!"
    
    # Assert that explicit query parameters have NO gradients!
    for name, p in model.supervised_attention.named_parameters():
        if p.grad is not None:
            assert p.grad.abs().sum().item() == 0.0, f"Supervised attention parameter {name} should have zero gradient in Phase 2!"
            
    print("🎉 Phase 2 gradient checks PASSED!")

    print("\n4. Phase 3 Simulation & Joint Fine-Tuning gradient flow check:")
    # Unfreeze absolutely everything
    model.unfreeze_backbone()
    model.unfreeze_supervised_attention()
    model.unfreeze_latent_attention()
    model.unfreeze_classifier()
    
    grad_params_p3 = [name for name, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters in Phase 3 (sample count: {len(grad_params_p3)}):")
    assert any("backbone" in name for name in grad_params_p3), "Backbone (LoRA) parameters should be trainable in Phase 3!"
    assert any("supervised_attention" in name for name in grad_params_p3), "Supervised attention parameters should be trainable in Phase 3!"
    assert any("latent_attention" in name for name in grad_params_p3), "Latent attention parameters should be trainable in Phase 3!"
    assert any("classifier_head" in name for name in grad_params_p3), "Classifier parameters should be trainable in Phase 3!"
    
    # Reset gradients
    model.zero_grad()
    
    # Joint forward pass
    class_logits, concept_logits, _, supervised_features, latent_features = model(x, return_features=True)
    
    # Compute Target Loss
    loss_target = F.cross_entropy(class_logits, dummy_targets)
    
    # Compute Concept Loss
    supervised_logits = concept_logits[:, :112]
    loss_concept = F.binary_cross_entropy_with_logits(supervised_logits, dummy_concepts)
    
    # Calculate Orthogonal Projection Loss
    explicit_norm_p3 = F.normalize(supervised_features, p=2, dim=-1)
    latent_norm_p3 = F.normalize(latent_features, p=2, dim=-1)
    cos_sim_p3 = torch.bmm(latent_norm_p3, explicit_norm_p3.transpose(1, 2))
    loss_latent_ortho_p3 = (cos_sim_p3 ** 2).mean()
    
    # Calculate Latent Sparsity Loss (L1)
    concept_probs_p3 = model.concept_activation(concept_logits)
    latent_activations_p3 = concept_probs_p3[:, 112:]
    loss_latent_l1_p3 = latent_activations_p3.abs().mean()
    
    # Combined joint loss
    total_loss_p3 = loss_target + 1.0 * loss_concept + 0.1 * loss_latent_ortho_p3 + 0.01 * loss_latent_l1_p3
    print("Joint loss computed:", total_loss_p3.item())
    
    total_loss_p3.backward()
    
    # Assert gradients are populated everywhere!
    backbone_grad_sum = sum(p.grad.abs().sum().item() for name, p in model.backbone.named_parameters() if p.grad is not None and "lora_" in name)
    supervised_grad_sum = sum(p.grad.abs().sum().item() for p in model.supervised_attention.parameters() if p.grad is not None)
    latent_grad_sum_p3 = sum(p.grad.abs().sum().item() for p in model.latent_attention.parameters() if p.grad is not None)
    classifier_grad_sum = sum(p.grad.abs().sum().item() for p in model.classifier_head.parameters() if p.grad is not None)
    
    print("Backbone LoRA gradients sum:", backbone_grad_sum)
    print("Supervised attention gradients sum:", supervised_grad_sum)
    print("Latent attention gradients sum:", latent_grad_sum_p3)
    print("Classifier head gradients sum:", classifier_grad_sum)
    
    assert backbone_grad_sum > 0.0, "Gradients should flow to Backbone adapters!"
    assert supervised_grad_sum > 0.0, "Gradients should flow to Supervised attention queries!"
    assert latent_grad_sum_p3 > 0.0, "Gradients should flow to Latent attention queries!"
    assert classifier_grad_sum > 0.0, "Gradients should flow to Classifier head parameters!"
    
    print("\n🎉 ALL PIPELINE DISENTANGLEMENT & GRADIENT ASSERTIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_latent_losses()
