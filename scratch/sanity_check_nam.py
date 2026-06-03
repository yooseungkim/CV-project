import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedSparseNAMHead(nn.Module):
    def __init__(self, num_concepts: int = 312, num_classes: int = 200, 
                 hidden_dim: int = 64, num_latent_concepts: int = 0):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_latent_concepts = num_latent_concepts
        
        # Non-linear Sub-networks (NAM) using Grouped Conv1D for parallelization
        # Maps [Batch, num_concepts, 1] -> [Batch, num_concepts * hidden_dim, 1]
        self.conv1 = nn.Conv1d(
            in_channels=num_concepts,
            out_channels=num_concepts * hidden_dim,
            kernel_size=1,
            groups=num_concepts
        )
        
        # Maps [Batch, num_concepts * hidden_dim, 1] -> [Batch, num_concepts * num_classes, 1]
        self.conv2 = nn.Conv1d(
            in_channels=num_concepts * hidden_dim,
            out_channels=num_concepts * num_classes,
            kernel_size=1,
            groups=num_concepts
        )
        
        # Learnable gating parameters initialized to 1.0 (all concepts active initially)
        self.concept_gates = nn.Parameter(torch.ones(num_concepts))
        
        # Linear layer for latent concepts if present
        if self.num_latent_concepts > 0:
            self.latent_linear = nn.Linear(num_latent_concepts, num_classes)
        else:
            self.latent_linear = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Split supervised and latent concepts
        supervised_x = x[:, :self.num_concepts]
        
        # Reshape to [Batch, num_concepts, 1] for Conv1D
        supervised_x = supervised_x.unsqueeze(-1)
        
        # Forward pass through NAM sub-networks
        # conv1: maps each concept's 1D logit to hidden_dim representation
        h = F.relu(self.conv1(supervised_x))
        # conv2: maps hidden_dim representation to num_classes output per concept
        y = self.conv2(h) # Shape: [Batch, num_concepts * num_classes, 1]
        
        # Reshape to [Batch, num_concepts, num_classes]
        y = y.view(batch_size, self.num_concepts, self.num_classes)
        
        # Apply learnable gates: gate[i] multiplies MLP_i output
        gated_y = y * self.concept_gates.view(1, self.num_concepts, 1)
        
        # Sum outputs of all gated non-linear concept models
        supervised_out = gated_y.sum(dim=1) # Shape: [Batch, num_classes]
        
        # Add latent concept representations if applicable
        if self.num_latent_concepts > 0 and self.latent_linear is not None:
            latent_x = x[:, self.num_concepts:]
            latent_out = self.latent_linear(latent_x)
            return supervised_out + latent_out
            
        return supervised_out

    def get_sparsity_loss(self) -> torch.Tensor:
        return torch.sum(torch.abs(self.concept_gates))

# Simple Sanity check
if __name__ == "__main__":
    batch_size = 16
    num_concepts = 312
    num_classes = 200
    num_latent = 20
    
    # Test with latent concepts
    x = torch.randn(batch_size, num_concepts + num_latent)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    head = GatedSparseNAMHead(
        num_concepts=num_concepts,
        num_classes=num_classes,
        hidden_dim=64,
        num_latent_concepts=num_latent
    )
    
    logits = head(x)
    print("Logits shape:", logits.shape)
    assert logits.shape == (batch_size, num_classes), "Incorrect logits shape"
    
    loss_fn = nn.CrossEntropyLoss()
    ce_loss = loss_fn(logits, targets)
    sparsity_loss = head.get_sparsity_loss()
    total_loss = ce_loss + 0.01 * sparsity_loss
    
    total_loss.backward()
    print("Backward pass succeeded!")
    print("Sparsity Loss:", sparsity_loss.item())
    print("Concept Gates Grad mean:", head.concept_gates.grad.abs().mean().item())
