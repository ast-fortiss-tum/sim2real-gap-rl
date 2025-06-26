import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt

""""
MAIN MODULE TO TRAIN DENOISER AUTOENCODER
This script trains an online recurrent denoising autoencoder on a dataset of noisy and clean episodes.
It uses a custom PyTorch Dataset to load the data, defines the model architecture,
and implements a training loop with early stopping and learning rate scheduling.
The model is designed to denoise sequences of observations from a manual control task.
The training process includes validation and saves the best model based on validation loss.
The model is then used to visualize the denoising performance on a few validation sequences.
"""

# ------------------------------
# 1. Create a PyTorch Dataset to Load the Saved Data
# ------------------------------
class DenoisingDataset(Dataset):
    def __init__(self, clean_file, noisy_file):
        # Load the datasets saved as .npy files (lists of episodes)
        self.clean_data = np.load(clean_file, allow_pickle=True)
        self.noisy_data = np.load(noisy_file, allow_pickle=True)
        print(self.clean_data.shape, self.noisy_data.shape)

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
        # Here we assume each episode must have a fixed length of 24
        assert x_clean.shape[0] == 24, f"Expected episode length 24, got {x_clean.shape[0]}"
        assert x_noisy.shape[0] == 24, f"Expected episode length 24, got {x_noisy.shape[0]}"
        # Return as torch tensors
        return torch.tensor(x_noisy), torch.tensor(x_clean)

# ------------------------------
# 2. Define the Online Recurrent Denoising Autoencoder
# ------------------------------
class OnlineDenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=1, proj_dim=16, lstm_hidden_dim=32, num_layers=1): #16,32
        super(OnlineDenoisingAutoencoder, self).__init__()
        self.input_linear = nn.Linear(input_dim, proj_dim)
        self.lstm = nn.LSTM(input_size=proj_dim, hidden_size=lstm_hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_linear = nn.Linear(lstm_hidden_dim, input_dim)
    
    def forward_online(self, x_seq, hidden=None):
        batch, T, _ = x_seq.size()
        outputs = []
        for t in range(T):
            x_t = x_seq[:, t, :]
            x_proj = self.input_linear(x_t)
            x_proj = x_proj.unsqueeze(1)
            out, hidden = self.lstm(x_proj, hidden)
            out = self.output_linear(out.squeeze(1))
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs, hidden

    def forward(self, x_seq):
        outputs, _ = self.forward_online(x_seq)
        return outputs

# ------------------------------
# 3. Prepare DataLoaders, Reproducibility, and Training Loop
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train online recurrent denoising autoencoder.")
    parser.add_argument("--noise", type=float, default=0.2, help="Gaussian noise std")
    parser.add_argument("--bias", type=float, default=0.5, help="Bias added to observation")
    parser.add_argument("--degree", type=float, default=0.65, help="Degree of the environment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # ─── Reproducibility Setup ─────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    # ──────────────────────────────────────────────────────────────

    base = "manual_control_dataset/"
    clean_file = f"{base}clean/clean_dataset_degree_{args.degree}_Gaussian_noise_{args.noise}_bias_{args.bias}.npy"
    noisy_file = f"{base}noisy/noisy_dataset_degree_{args.degree}_Gaussian_noise_{args.noise}_bias_{args.bias}.npy"

    #base = "shift_dataset/"
    #clean_file = f"{base}clean_shift/clean_dataset_degree_{args.degree}_Gaussian_noise_{args.noise}_bias_{args.bias}.npy"
    #noisy_file = f"{base}noisy_shift/noisy_dataset_degree_{args.degree}_Gaussian_noise_{args.noise}_bias_{args.bias}.npy"

    full_dataset = DenoisingDataset(clean_file, noisy_file)
    total_samples = len(full_dataset)

    # Train/validation split
    train_ratio = 0.8
    train_size = int(total_samples * train_ratio)
    val_size   = total_samples - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    print(f"Total: {total_samples}, Train: {train_size}, Val: {val_size}")

    batch_size = 24
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, generator=generator)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, generator=generator)

    # Model, loss, optimizer, scheduler
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = OnlineDenoisingAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop with early stopping
    num_epochs = 1000
    patience   = 20
    best_val   = float('inf')
    no_improve = 0

    for epoch in range(1, num_epochs+1):
        # Training
        model.train()
        train_loss = 0.0
        for x_noisy, x_clean in train_loader:
            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)
            optimizer.zero_grad()
            recon, _ = model.forward_online(x_noisy)
            loss = criterion(recon, x_clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_noisy, x_clean in val_loader:
                x_noisy = x_noisy.to(device)
                x_clean = x_clean.to(device)
                recon, _ = model.forward_online(x_noisy)
                val_loss += criterion(recon, x_clean).item()
        avg_val = val_loss / len(val_loader)

        print(f"Epoch {epoch}/{num_epochs} - Train: {avg_train:.6f}, Val: {avg_val:.6f}")
        scheduler.step(avg_val)

        # Early stopping check
        path = f"Denoising_AE/best_online_noise_{args.noise}_bias_{args.bias}_deg_{args.degree}.pth"
        if avg_val < best_val:
            best_val = avg_val
            no_improve = 0
            torch.save(model.state_dict(), path)
            print("  Improvement, model saved.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    # Load best model for visual inspection
    model.load_state_dict(torch.load(path, map_location=device))
    print("Loaded best model.")

    # Plot a few validation sequences
    model.eval()
    with torch.no_grad():
        for x_noisy, x_clean in val_loader:
            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)
            recon, _ = model.forward_online(x_noisy)
            x_noisy = x_noisy.cpu().numpy()
            x_clean = x_clean.cpu().numpy()
            recon   = recon.cpu().numpy()
            T = x_noisy.shape[1]
            for i in range(min(x_noisy.shape[0], 4)):
                plt.figure(figsize=(10,4))
                t = np.arange(1, T+1)
                plt.plot(t, x_noisy[i].squeeze(), 'r--', label='Noisy')
                plt.plot(t, recon[i].squeeze(),   'b-',  label='Denoised')
                plt.plot(t, x_clean[i].squeeze(), 'g:',  label='Clean')
                plt.xlabel("Timestep")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            break

if __name__ == "__main__":
    main()
