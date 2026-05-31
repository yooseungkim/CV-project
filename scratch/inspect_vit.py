import timm
import torch

try:
    model = timm.create_model("vit_base_patch14_dinov2", pretrained=False)
    print("Blocks count:", len(model.blocks))
    print("Final block attn class:", model.blocks[-1].attn.__class__)
    print("Final block attn modules and parameters:")
    for name, module in model.blocks[-1].attn.named_modules():
        print(f"  {name}: {module.__class__}")
except Exception as e:
    print("Error:", e)
