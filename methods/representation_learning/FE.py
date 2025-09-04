import torch
import torch.nn as nn

# Example model that uses the embedding tensor
class FE(nn.Module):
    def __init__(self, num_nodes=356, embedding_d=50):
        super(FE, self).__init__()
        self.embedding = torch.nn.Parameter(torch.randn(num_nodes, embedding_d))

    def forward(self):
        # Example forward pass using the embedding tensor
        embedded = self.embedding
        return embedded