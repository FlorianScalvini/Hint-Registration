import torch
import numpy as np
import matplotlib.pyplot as plt

# Simulate active learning process where loss is randomly generated at each step
epochs = torch.arange(1, 101)  # 100 training epochs (converted to tensor)
loss_values = torch.rand(100)  # Random loss values between 0 and 1 for each epoch (tensor)

# Compute cumulative loss
cumulative_loss = torch.cumsum(loss_values, dim=0)

# Normalize the cumulative loss between 0 and 1
cumulative_loss_normalized = (cumulative_loss - torch.min(cumulative_loss)) / (torch.max(cumulative_loss) - torch.min(cumulative_loss))

# Find the nearest index where y = 0.3
target_value = 0.3
nearest_index = torch.abs(cumulative_loss_normalized - target_value).argmin()

# Print the nearest index and corresponding epoch
print(f"Nearest index to y = {target_value} is at epoch {epochs[nearest_index].item()}, with normalized loss {cumulative_loss_normalized[nearest_index].item():.4f}")

# Plot the normalized cumulative loss
plt.figure(figsize=(8, 6))
plt.plot(epochs.numpy(), cumulative_loss_normalized.numpy(), label="Normalized Cumulative Loss", color='r', linewidth=2)
plt.scatter(epochs[nearest_index].item(), cumulative_loss_normalized[nearest_index].item(), color='blue', zorder=5, label=f'Nearest Index (y=0.3)')
plt.title("Normalized Cumulative Loss in Active Learning over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Normalized Cumulative Loss")
plt.grid(True)
plt.legend(loc="best")
plt.show()


