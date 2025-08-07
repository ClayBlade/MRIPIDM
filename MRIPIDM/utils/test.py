import torch
import numpy as np

# List to store all individual matrices
matrices = []

for _ in range(10):  
    matrix = torch.randn(16, 16)       # generate a 16x16 matrix with random values
    matrix = matrix.unsqueeze(0)       # add channel dimension: (1, 16, 16)
    matrices.append(matrix)

# Stack into a single tensor of shape (10, 1, 16, 16)
batch_tensor = torch.stack(matrices)

print(batch_tensor.shape)  # Output: torch.Size([10, 1, 16, 16])


