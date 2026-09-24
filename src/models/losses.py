import torch
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: Tensors of shape (batch_size, feature_dim)
        """
        batch_size = z_i.shape[0]
        
        # 1. Normalize embeddings along feature dimension
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # 2. Concatenate all embeddings: Shape (2 * batch_size, feature_dim)
        representations = torch.cat([z_i, z_j], dim=0)
        
        # 3. Compute pairwise Cosine Similarity Matrix: Shape (2N, 2N)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # 4. Mask out self-contrast (diagonal entries)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix.masked_fill_(mask, -9e15)
        
        # 5. Build ground truth target indices for Cross Entropy
        # For item k (from z_i), its positive pair is k + batch_size (in z_j)
        # For item k + batch_size (from z_j), its positive pair is k (in z_i)
        targets = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z_i.device),
            torch.arange(0, batch_size, device=z_i.device)
        ], dim=0)
        
        # 6. Apply standard Cross Entropy Loss over similarity scores
        loss = F.cross_entropy(similarity_matrix, targets)
        return loss