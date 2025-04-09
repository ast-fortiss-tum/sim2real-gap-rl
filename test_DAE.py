import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Assume your previous code (dataset, model, etc.) has already been defined and the best model is saved as 'best_denoising_autoencoder.pth'

# Load the validation dataset (or use the full dataset for testing).
dataset_path_clean = 'clean_dataset.npy'
dataset_path_noisy = 'noisy_dataset.npy'
# Here, we use the same DenoisingDataset from before.
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
        # Here we assume each episode is 24 timesteps.
        assert x_clean.shape[0] == 25, f"Episode length expected 24, got {x_clean.shape[0]}"
        assert x_noisy.shape[0] == 25, f"Episode length expected 24, got {x_noisy.shape[0]}"
        return torch.tensor(x_noisy), torch.tensor(x_clean)

# Create a dataset and DataLoader for visualization.
test_dataset = DenoisingDataset(dataset_path_clean, dataset_path_noisy)
# For visualization, you might want to work on a small batch.
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

# Instantiate the model (architecture must be the same as used in training)
# Assuming the DenoisingAutoencoder definition from before.
class DenoisingAutoencoder(nn.Module):
    def __init__(self, cnn_channels=16, lstm_hidden_dim=32, decoder_hidden_dim=32, num_layers=1):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.encoder_lstm = nn.LSTM(input_size=cnn_channels,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    batch_first=True)
        self.decoder_lstm = nn.LSTM(input_size=lstm_hidden_dim,
                                    hidden_size=decoder_hidden_dim,
                                    num_layers=num_layers,
                                    batch_first=True)
        self.decoder_fc = nn.Linear(decoder_hidden_dim, 1)
        
    def forward(self, x):
        batch_size, T, _ = x.size()  # T should be 24
        x_cnn = x.transpose(1,2)  # (batch, 1, 24)
        cnn_out = self.encoder_cnn(x_cnn)  # (batch, cnn_channels, 12)
        cnn_out = cnn_out.transpose(1,2)   # (batch, 12, cnn_channels)
        lstm_out, (h, c) = self.encoder_lstm(cnn_out)
        latent = h[-1]  # (batch, lstm_hidden_dim)
        T_down = cnn_out.size(1)  # 12
        latent_repeated = latent.unsqueeze(1).repeat(1, T_down, 1)
        dec_out, _ = self.decoder_lstm(latent_repeated)
        recon_intermediate = self.decoder_fc(dec_out)  # (batch, 12, 1)
        recon = torch.nn.functional.interpolate(recon_intermediate.transpose(1,2), size=T, mode='linear', align_corners=True).transpose(1,2)
        return recon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingAutoencoder(cnn_channels=16, lstm_hidden_dim=32, decoder_hidden_dim=32, num_layers=1)
model = model.to(device)

# Load the best model weights.
model.load_state_dict(torch.load("best_denoising_autoencoder.pth", map_location=device))
model.eval()

# Visualize a few examples.
with torch.no_grad():
    for batch_idx, (x_noisy, x_clean) in enumerate(test_loader):
        x_noisy = x_noisy.to(device)
        x_clean = x_clean.to(device)
        recon = model(x_noisy)
        # Move to CPU for plotting.
        x_noisy = x_noisy.cpu().numpy()
        x_clean = x_clean.cpu().numpy()
        recon = recon.cpu().numpy()
        
        # For each sample in the batch, plot the three curves.
        for i in range(x_noisy.shape[0]):
            plt.figure(figsize=(10, 4))
            # Each observation sequence is of shape (24, 1). Squeeze to get (24,)
            noisy_seq = x_noisy[i].squeeze()
            clean_seq = x_clean[i].squeeze()
            recon_seq = recon[i].squeeze()
            timesteps = np.arange(0, 25)
            plt.plot(timesteps, noisy_seq, 'r--', label='Noisy Input')
            plt.plot(timesteps, recon_seq, 'b-', linewidth=2, label='Denoised (Reconstruction)')
            plt.plot(timesteps, clean_seq, 'g:', linewidth=2, label='Clean Target')
            plt.xlabel("Timestep")
            plt.ylabel("Observation Value")
            plt.title(f"Example {batch_idx * test_loader.batch_size + i + 1}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        # For demonstration, we'll visualize just one batch.
        break
