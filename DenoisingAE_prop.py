import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
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
        # Here we assume each episode must have a fixed length of 24 timesteps.
        assert x_clean.shape[0] == 25, f"Expected episode length 24, got {x_clean.shape[0]}"
        assert x_noisy.shape[0] == 25, f"Expected episode length 24, got {x_noisy.shape[0]}"
        # Return as torch tensors
        return torch.tensor(x_noisy), torch.tensor(x_clean)

# ------------------------------
# 2. Define the CNN+LSTM Denoising Autoencoder
# ------------------------------
class DenoisingAutoencoder(nn.Module):
    def __init__(self, cnn_channels=16, lstm_hidden_dim=32, decoder_hidden_dim=32, num_layers=1):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder CNN: Input channels=1, output channels=cnn_channels.
        # Uses kernel_size=3, padding=1 to preserve sequence length then max-pooling.
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Downsamples from 24 to 12 timesteps.
        )
        # Encoder LSTM: Input size equals cnn_channels.
        self.encoder_lstm = nn.LSTM(input_size=cnn_channels,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    batch_first=True)
        # Decoder:
        # Repeat the latent vector and decode using an LSTM.
        self.decoder_lstm = nn.LSTM(input_size=lstm_hidden_dim,
                                    hidden_size=decoder_hidden_dim,
                                    num_layers=num_layers,
                                    batch_first=True)
        # Final reconstruction layer maps the decoder's hidden state to output (1 dim).
        self.decoder_fc = nn.Linear(decoder_hidden_dim, 1)
        
    def forward(self, x):
        """
        Input: x of shape (batch, T, 1) with T=24.
        Output: Reconstruction of shape (batch, T, 1).
        """
        batch_size, T, _ = x.size()
        # --- Encoder ---
        # CNN expects (batch, channels, T)
        x_cnn = x.transpose(1, 2)  # Now (batch, 1, 24)
        cnn_out = self.encoder_cnn(x_cnn)  # (batch, cnn_channels, T_down), where T_down=12
        cnn_out = cnn_out.transpose(1, 2)   # (batch, 12, cnn_channels)
        # Pass through encoder LSTM.
        lstm_out, (h, c) = self.encoder_lstm(cnn_out)
        latent = h[-1]  # (batch, lstm_hidden_dim)
        
        # --- Decoder ---
        # Use T_down (12) as the sequence length for the decoder.
        T_down = cnn_out.size(1)
        latent_repeated = latent.unsqueeze(1).repeat(1, T_down, 1)  # (batch, 12, lstm_hidden_dim)
        dec_out, _ = self.decoder_lstm(latent_repeated)  # (batch, 12, decoder_hidden_dim)
        recon_intermediate = self.decoder_fc(dec_out)  # (batch, 12, 1)
        # Upsample reconstruction from 12 timesteps back to 24 using linear interpolation.
        recon = F.interpolate(recon_intermediate.transpose(1, 2), size=T, mode='linear', align_corners=True).transpose(1,2)
        return recon  # (batch, 24, 1)

# ------------------------------
# 3. Prepare the Dataset, Split Into Train and Validation, Create DataLoaders
# ------------------------------
dataset_path_clean = 'clean_dataset.npy'
dataset_path_noisy = 'noisy_dataset.npy'

full_dataset = DenoisingDataset(dataset_path_clean, dataset_path_noisy)
total_samples = len(full_dataset)

# Define a split ratio, e.g., 80% for training and 20% for validation.
train_ratio = 0.8
train_size = int(total_samples * train_ratio)
val_size = total_samples - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print(f"Total samples: {total_samples}, Training: {train_size}, Validation: {val_size}")

batch_size = 25
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------
# 4. Set Up the Training with Early Stopping and LR Scheduler
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingAutoencoder(cnn_channels=16, lstm_hidden_dim=32, decoder_hidden_dim=32, num_layers=1)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# Use ReduceLROnPlateau: if validation loss does not improve for 10 epochs, reduce lr by a factor of 0.5.
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

num_epochs = 1000        # Maximum number of epochs
patience = 20           # Early stopping patience: stop if no improvement for 20 epochs
best_val_loss = float('inf')
epochs_no_improve = 0

print("Starting training with early stopping and LR scheduler...")
for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train()
    train_loss = 0.0
    for batch_idx, (x_noisy, x_clean) in enumerate(train_loader):
        x_noisy = x_noisy.to(device)   # (batch, 24, 1)
        x_clean = x_clean.to(device)
        optimizer.zero_grad()
        recon = model(x_noisy)
        loss = criterion(recon, x_clean)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    
    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_noisy, x_clean in val_loader:
            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)
            recon = model(x_noisy)
            loss = criterion(recon, x_clean)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")
    
    # Step the LR scheduler using the validation loss.
    scheduler.step(avg_val_loss)
    
    # --- Early Stopping Check ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_denoising_autoencoder.pth")
        print("  Validation loss improved. Model saved.")
    else:
        epochs_no_improve += 1
        print(f"  No improvement for {epochs_no_improve} epoch(s).")
    
    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        break

print("Training complete.")

# Load the best model (optional)
model.load_state_dict(torch.load("best_denoising_autoencoder.pth"))
print("Best model loaded.")
