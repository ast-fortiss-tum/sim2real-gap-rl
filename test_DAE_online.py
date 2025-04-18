import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------
# Assume this is your dataset class.
# -------------------------------------
class DenoisingDataset(torch.utils.data.Dataset):
    def __init__(self, clean_file, noisy_file):
        self.clean_data = np.load(clean_file, allow_pickle=True)
        self.noisy_data = np.load(noisy_file, allow_pickle=True)
        self.n = len(self.clean_data)
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        x_clean = self.clean_data[idx]
        x_noisy = self.noisy_data[idx]
        if len(x_clean.shape) == 1:
            x_clean = np.expand_dims(x_clean, -1)
        if len(x_noisy.shape) == 1:
            x_noisy = np.expand_dims(x_noisy, -1)
        x_clean = x_clean.astype(np.float32)
        x_noisy = x_noisy.astype(np.float32)
        # Here we assume each episode has a fixed length of 25 timesteps.
        assert x_clean.shape[0] == 24, f"Episode length expected 24, got {x_clean.shape[0]}"
        assert x_noisy.shape[0] == 24, f"Episode length expected 24, got {x_noisy.shape[0]}"
        return torch.tensor(x_noisy), torch.tensor(x_clean)

# -------------------------------------
# Define your online denoising autoencoder.
# This model is supposed to process a sequence online.
# -------------------------------------
class OnlineDenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=1, proj_dim=16, lstm_hidden_dim=32, num_layers=1):
        super(OnlineDenoisingAutoencoder, self).__init__()
        self.input_linear = nn.Linear(input_dim, proj_dim)
        self.lstm = nn.LSTM(input_size=proj_dim, hidden_size=lstm_hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.output_linear = nn.Linear(lstm_hidden_dim, input_dim)
        
    def forward_online(self, x_seq, hidden=None):
        # x_seq: (batch, T, input_dim) where T >= 1.
        batch, T, _ = x_seq.size()
        outputs = []
        for t in range(T):
            x_t = x_seq[:, t, :]             # (batch, input_dim)
            x_proj = self.input_linear(x_t)    # (batch, proj_dim)
            x_proj = x_proj.unsqueeze(1)       # (batch, 1, proj_dim)
            out, hidden = self.lstm(x_proj, hidden)  # (batch, 1, lstm_hidden_dim)
            out = self.output_linear(out.squeeze(1)) # (batch, input_dim)
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # (batch, T, input_dim)
        return outputs, hidden

    def forward(self, x_seq):
        outputs, _ = self.forward_online(x_seq)
        return outputs

# -------------------------------------
# Set device and load the pretrained online denoiser.
# -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
denoiser = OnlineDenoisingAutoencoder(input_dim=1, proj_dim=16, lstm_hidden_dim=32, num_layers=1).to(device)
denoiser.load_state_dict(torch.load("Denoising_AE/best_online_denoising_autoencoder.pth", map_location=device))
denoiser.eval()

# -------------------------------------
# Load your dataset.
# -------------------------------------
dataset_path_clean = 'clean_dataset.npy'
dataset_path_noisy = 'noisy_dataset.npy'
test_dataset = DenoisingDataset(dataset_path_clean, dataset_path_noisy)
# For testing/demonstration, we take one sample from the dataset.
# You can use DataLoader for batching, but here we directly index.
noisy_sample, clean_sample = test_dataset[145]
# Convert to torch tensor and add batch dimension.
# Both have shape (T, 1) where T is 25.
noisy_sample = noisy_sample.unsqueeze(0).to(device)   # shape: (1, 25, 1)
clean_sample = clean_sample.unsqueeze(0).to(device)   # shape: (1, 25, 1)

# -------------------------------------
# Online Denoising Test over an accumulating buffer.
# For each t = 1,2,...,25, we create a window from the first t timesteps,
# pass it through the denoiser, and record the recovered value at the last timestep.
# -------------------------------------
recovered_values = []
window_lengths = []

T_total = noisy_sample.shape[1]  # Should be 25.
for t in range(1, T_total + 1):
    current_window = noisy_sample[:, :t, :]  # shape: (1, t, 1)
    with torch.no_grad():
        denoised_seq, _ = denoiser.forward_online(current_window)
    # recovered value from the last timestep in the window:
    recovered_value = denoised_seq[0, -1, :].item()  # scalar value
    recovered_values.append(recovered_value)
    window_lengths.append(t)

# Extract the corresponding clean target values for comparison.
# We assume that the clean sample is the ideal (denoised) target.
# For each t, the true value is simply clean_sample[0, t-1, 0].
true_values = [clean_sample[0, t-1, 0].item() for t in range(1, T_total + 1)]

# Extract also the original noisy values for reference.
noisy_values = [noisy_sample[0, t-1, 0].item() for t in range(1, T_total + 1)]

# -------------------------------------
# Plot the evolution of the recovered value with the increasing window.
# -------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(window_lengths, noisy_values, 'r--', marker='o', label='Original Noisy Value (at t)')
plt.plot(window_lengths, recovered_values, 'b-', marker='x', label='Recovered Value (with window of t)')
plt.plot(window_lengths, true_values, 'g:', marker='s', label='Clean Target')
plt.xlabel("Window Length (Number of samples accumulated)")
plt.ylabel("Value at the last timestep")
plt.title("Online Denoising: Recovered Value vs. Clean Target vs. Noisy Input")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
