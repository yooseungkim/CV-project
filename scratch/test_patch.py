import timm
import torch
import types

model = timm.create_model("vit_base_patch14_dinov2", pretrained=False, dynamic_img_size=True)
attn_module = model.blocks[-1].attn

# Save the original forward just in case
attn_module.fused_attn = False

def custom_forward(self, x, attn_mask=None, is_causal=False):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    
    q = q * self.scale
    attn = q @ k.transpose(-2, -1)
    
    if attn_mask is not None:
        from timm.layers.attention import resolve_self_attn_mask, maybe_add_mask
        attn_bias = resolve_self_attn_mask(N, attn, attn_mask, is_causal)
        attn = maybe_add_mask(attn, attn_bias)
        
    attn = attn.softmax(dim=-1)
    self.last_attn_weights = attn
    
    attn = self.attn_drop(attn)
    x = attn @ v
    
    x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
    x = self.norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

attn_module.forward = types.MethodType(custom_forward, attn_module)

# Pass dummy input
dummy_x = torch.randn(2, 3, 224, 224)
out = model.forward_features(dummy_x)
print("Output shape:", out.shape if not isinstance(out, tuple) else out[0].shape)
print("Captured attention weights shape:", getattr(attn_module, "last_attn_weights", None).shape)
