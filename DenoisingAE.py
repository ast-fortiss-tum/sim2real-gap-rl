import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ------------------------------
# 1. Create a PyTorch Dataset to Load the Saved Data
# ------------------------------
class DenoisingDataset(Dataset):
    def __init__(self, clean_file, noisy_file):
        # Load the datasets saved as .npy files (lists of episodes)
        self.clean_data = np.load(clean_file, allow_pickle=True)
        self.noisy_data = np.load(noisy_file, allow_pickle=True)
        self.n = len(self.clean_data)
        
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        # Get episode from clean and noisy dataset
        x_clean = self.clean_data[idx]
        x_noisy = self.noisy_data[idx]
        # If an episode is 1D, expand dims to get (T, 1)
        if len(x_clean.shape) == 1:
            x_clean = np.expand_dims(x_clean, -1)
        if len(x_noisy.shape) == 1:
            x_noisy = np.expand_dims(x_noisy, -1)
        # Ensure float32 types
        x_clean = x_clean.astype(np.float32)
        x_noisy = x_noisy.astype(np.float32)
        # Return as torch tensors
        return torch.tensor(x_noisy), torch.tensor(x_clean)

# ------------------------------
# 2. Define the CNN+LSTM Denoising Autoencoder
# ------------------------------
class DenoisingAutoencoder(nn.Module):
    def __init__(self, cnn_channels=16, lstm_hidden_dim=32, decoder_hidden_dim=32, num_layers=1):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder CNN: Input channels=1, output channels=cnn_channels.
        # We use a kernel_size=3, padding=1 to preserve sequence length and then a max-pooling
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Downsamples sequence length by 2.
        )
        # Encoder LSTM: input size equals cnn_channels.
        self.encoder_lstm = nn.LSTM(input_size=cnn_channels,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    batch_first=True)
        # Decoder:
        # The decoder LSTM will use the latent vector to generate a sequence.
        # Here, we simply repeat the latent vector to form an initial sequence.
        self.decoder_lstm = nn.LSTM(input_size=lstm_hidden_dim,
                                    hidden_size=decoder_hidden_dim,
                                    num_layers=num_layers,
                                    batch_first=True)
        # Final reconstruction layer maps decoder hidden state to output dimension (1).
        self.decoder_fc = nn.Linear(decoder_hidden_dim, 1)
        
    def forward(self, x):
        """
        Input: x of shape (batch, T, 1)
        Output: reconstruction of shape (batch, T, 1)
        """
        batch_size, T, _ = x.size()
        # --- Encoder ---
        # CNN expects input shape (batch, channels, T)
        x_cnn = x.transpose(1, 2)  # Now (batch, 1, T)
        cnn_out = self.encoder_cnn(x_cnn)  # (batch, cnn_channels, T_down), where T_down ~ T/2
        # Transpose back to (batch, T_down, cnn_channels)
        cnn_out = cnn_out.transpose(1, 2)
        # Pass through encoder LSTM
        lstm_out, (h, c) = self.encoder_lstm(cnn_out)
        # Take final hidden state from last layer as latent representation: shape (batch, lstm_hidden_dim)
        latent = h[-1]
        
        # --- Decoder ---
        # Decide on the sequence length to generate.
        # Here we use T_down, the length after CNN pooling.
        T_down = cnn_out.size(1)
        # Repeat the latent vector T_down times to form an initial input for the decoder.
        latent_repeated = latent.unsqueeze(1).repeat(1, T_down, 1)
        # Pass through decoder LSTM.
        dec_out, _ = self.decoder_lstm(latent_repeated)
        # Map each time step through the FC to get the output.
        recon_intermediate = self.decoder_fc(dec_out)  # (batch, T_down, 1)
        # Upsample the reconstruction to match the original sequence length T.
        # We use linear interpolation.
        recon = F.interpolate(recon_intermediate.transpose(1,2), size=T, mode='linear', align_corners=True).transpose(1,2)
        # recon shape: (batch, T, 1)
        return recon

# ------------------------------
# 3. Load the Saved Dataset and Create a DataLoader
# ------------------------------
dataset_path_clean = 'clean_dataset.npy'
dataset_path_noisy = 'noisy_dataset.npy'

denoise_dataset = DenoisingDataset(dataset_path_clean, dataset_path_noisy)
dataloader = DataLoader(denoise_dataset, batch_size=16, shuffle=True)

# ------------------------------
# 4. Train the Autoencoder
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingAutoencoder(cnn_channels=16, lstm_hidden_dim=32, decoder_hidden_dim=32, num_layers=1).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 200
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (x_noisy, x_clean) in enumerate(dataloader):
        x_noisy = x_noisy.to(device)   # shape: (batch, T, 1)
        x_clean = x_clean.to(device)
        optimizer.zero_grad()
        recon = model(x_noisy)         # Output shape: (batch, T, 1)
        loss = criterion(recon, x_clean)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
    
# Save the trained model.
torch.save(model.state_dict(), "denoising_autoencoder.pth")
print("Autoencoder trained and saved as 'denoising_autoencoder.pth'.")
