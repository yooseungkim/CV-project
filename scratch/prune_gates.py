import torch
import os
import glob

def prune_checkpoint(path):
    print(f"\nProcessing checkpoint: {path}")
    try:
        state_dict = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint {path}: {e}")
        return

    # Check if this is a checkpoint dictionary containing state_dict, or just a state_dict
    sd = state_dict
    is_wrapped = False
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        sd = state_dict["state_dict"]
        is_wrapped = True

    gate_key = None
    for k in sd.keys():
        if "classifier_head.concept_gates" in k:
            gate_key = k
            break

    if gate_key is None:
        print("No concept gates found in this checkpoint.")
        return

    gates = sd[gate_key]
    total_gates = gates.numel()
    
    # Original active gates
    original_active = (gates.abs() > 0.0).sum().item()
    original_active_005 = (gates.abs() > 0.05).sum().item()
    
    # Prune
    pruned_gates = gates.clone()
    pruned_gates[pruned_gates.abs() <= 0.05] = 0.0
    final_active = (pruned_gates.abs() > 0.0).sum().item()

    print(f"Total Gates: {total_gates}")
    print(f"Original active (>0.0): {original_active}")
    print(f"Original active (>0.05): {original_active_005}")
    print(f"Final remaining gates (after pruning <= 0.05): {final_active}")

    # Overwrite/save pruned gates back if needed, but let's first report the counts.
    # Update the state dict in place
    sd[gate_key] = pruned_gates
    
    # Save a pruned copy next to the original
    base, ext = os.path.splitext(path)
    pruned_path = f"{base}_pruned{ext}"
    torch.save(state_dict, pruned_path)
    print(f"Saved pruned checkpoint to: {pruned_path}")

if __name__ == "__main__":
    # Find all .pt and .pth checkpoints
    checkpoints = []
    for root, dirs, files in os.walk("checkpoints"):
        for file in files:
            if file.endswith((".pt", ".pth")) and "_pruned" not in file:
                checkpoints.append(os.path.join(root, file))
                
    print(f"Found {len(checkpoints)} checkpoints.")
    for cp in sorted(checkpoints):
        # We can prune all of them or select the latest/relevant ones
        if "latent" in cp:
            prune_checkpoint(cp)
