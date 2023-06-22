import torch
import matplotlib.pyplot as plt

# Define the tensor
tensor = torch.Tensor([[1.1, 1.3, 1.2, 1.1, 1.3]])

# Calculate mean and standard deviation
mean = tensor.mean().item()
std = tensor.std().item()

# Plot the tensor data with error bars
plt.errorbar(range(tensor.size(1)), tensor.flatten(), yerr=std, marker='o', linestyle='-', label='Data')
plt.axhline(mean, color='r', linestyle='--', label='Mean')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()