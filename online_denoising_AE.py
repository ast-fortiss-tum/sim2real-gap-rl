import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import argparse

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
        # Here we assume each episode must have a fixed length.
        # (Adjust the expected length if needed; here it is set to 25.)
        assert x_clean.shape[0] == 24, f"Expected episode length 25, got {x_clean.shape[0]}"
        assert x_noisy.shape[0] == 24, f"Expected episode length 25, got {x_noisy.shape[0]}"
        # Return as torch tensors
        return torch.tensor(x_noisy), torch.tensor(x_clean)

# ------------------------------
# 2. Define the Online Recurrent Denoising Autoencoder
# ------------------------------
class OnlineDenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim=1, proj_dim=16, lstm_hidden_dim=32, num_layers=1):
        """
        Online autoencoder that processes input one time step at a time.
        - Projects the input using a linear layer.
        - Uses an LSTM to maintain a hidden state over time.
        - Reconstructs the input from the LSTM output using another linear layer.
        
        Args:
            input_dim (int): Dimension of the input sample (default: 1).
            proj_dim (int): Dimension of the projected input.
            lstm_hidden_dim (int): Hidden dimension of the LSTM.
            num_layers (int): Number of LSTM layers.
        """
        super(OnlineDenoisingAutoencoder, self).__init__()
        self.input_linear = nn.Linear(input_dim, proj_dim)
        self.lstm = nn.LSTM(input_size=proj_dim, hidden_size=lstm_hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_linear = nn.Linear(lstm_hidden_dim, input_dim)
    
    def forward_online(self, x_seq, hidden=None):
        """
        Unrolls the sequence online (one time step at a time).
        
        Args:
            x_seq (torch.Tensor): Input sequence of shape (batch, T, input_dim)
            hidden (tuple or None): Hidden state for the LSTM (h, c). If None, initialized as zeros.
            
        Returns:
            outputs (torch.Tensor): Reconstructed output, shape (batch, T, input_dim)
            hidden (tuple): Final LSTM hidden state.
        """
        batch, T, _ = x_seq.size()
        outputs = []
        # Process each time step sequentially
        for t in range(T):
            # Extract the t-th timestep (shape: (batch, input_dim))
            x_t = x_seq[:, t, :]  
            # Project input: shape -> (batch, proj_dim)
            x_proj = self.input_linear(x_t)
            # Add a time dimension for the LSTM: (batch, 1, proj_dim)
            x_proj = x_proj.unsqueeze(1)
            # Process through LSTM one time step
            out, hidden = self.lstm(x_proj, hidden)  # out: (batch, 1, lstm_hidden_dim)
            # Reconstruct output: shape -> (batch, input_dim)
            out = self.output_linear(out.squeeze(1))
            # Append the output (keep time dimension)
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # shape: (batch, T, input_dim)
        return outputs, hidden

    def forward(self, x_seq):
        """
        Convenience method to process a full sequence online.
        """
        outputs, _ = self.forward_online(x_seq)
        return outputs

# ------------------------------
# 3. Prepare the Dataset, Split Into Train and Validation, Create DataLoaders
# ------------------------------

def main():

    parser = argparse.ArgumentParser(description="Collect paired clean and noisy datasets.")
    parser.add_argument("--noise", type=float, default=0.2, help="Standard deviation of Gaussian noise")
    parser.add_argument("--bias", type=float, default=0.5, help="Bias added to observation")
    parser.add_argument("--degree", type=float, default=0.65, help="Degree of the environment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment")
    args = parser.parse_args()

    base = "manual_control_dataset/"

    dataset_path_clean = base + f'clean/clean_dataset_degree_{args.degree}_Gaussian_noise_{args.noise}_bias_{args.bias}.npy'
    dataset_path_noisy = base + f'noisy/noisy_dataset_degree_{args.degree}_Gaussian_noise_{args.noise}_bias_{args.bias}.npy'

    full_dataset = DenoisingDataset(dataset_path_clean, dataset_path_noisy)
    total_samples = len(full_dataset)

    # Define a split ratio, e.g., 80% for training and 20% for validation.
    train_ratio = 0.8
    train_size = int(total_samples * train_ratio)
    val_size = total_samples - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Total samples: {total_samples}, Training: {train_size}, Validation: {val_size}")

    batch_size = 24  # Adjust batch size as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------------------------------
    # 4. Set Up the Training with Early Stopping and LR Scheduler for Online Model
    # ------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OnlineDenoisingAutoencoder(input_dim=1, proj_dim=16, lstm_hidden_dim=32, num_layers=1)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    num_epochs = 1000   # Maximum epochs
    patience = 20       # Early stopping patience
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("Starting online training with early stopping and LR scheduler...")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        for batch_idx, (x_noisy, x_clean) in enumerate(train_loader):
            # x_noisy, x_clean: (batch, T, 1) where T=25
            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)
            optimizer.zero_grad()
            # Process online: unroll the sequence one step at a time
            recon, _ = model.forward_online(x_noisy)
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
                recon, _ = model.forward_online(x_noisy)
                loss = criterion(recon, x_clean)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {avg_train_loss:.5f}  Val Loss: {avg_val_loss:.5f}")
        
        # Update the LR scheduler with the validation loss.
        scheduler.step(avg_val_loss)

        name = f"Denoising_AE/best_online_denoising_autoencoder_Gaussian_Noise_{args.noise}_Bias_{args.bias}_Degree_{args.degree}.pth"
        
        # --- Early Stopping Check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), name)
            print("  Validation loss improved. Model saved.")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s).")
        
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print("Online training complete.")

    # Load the best model (optional)
    model.load_state_dict(torch.load(name, map_location=device))
    print("Best online model loaded.")

    # ------------------------------
    # 5. Visual Testing of the Online Denoising
    # ------------------------------
    import matplotlib.pyplot as plt

    model.eval()
    # Get one batch from the validation loader
    with torch.no_grad():
        for x_noisy, x_clean in val_loader:
            x_noisy = x_noisy.to(device)
            x_clean = x_clean.to(device)
            recon, _ = model.forward_online(x_noisy)
            # Bring data back to CPU and convert to numpy arrays.
            x_noisy = x_noisy.cpu().numpy()
            x_clean = x_clean.cpu().numpy()
            recon = recon.cpu().numpy()
            # Plot each sample in the batch.
            for i in range(x_noisy.shape[0]):
                plt.figure(figsize=(10, 4))
                # Each sequence is of shape (T, 1); squeeze to (T,)
                noisy_seq = x_noisy[i].squeeze()
                clean_seq = x_clean[i].squeeze()
                recon_seq = recon[i].squeeze()
                timesteps = np.arange(1, x_noisy.shape[1] + 1)
                plt.plot(timesteps, noisy_seq, 'r--', label='Noisy Input')
                plt.plot(timesteps, recon_seq, 'b-', linewidth=2, label='Denoised Output')
                plt.plot(timesteps, clean_seq, 'g:', linewidth=2, label='Clean Target')
                plt.xlabel("Timestep")
                plt.ylabel("Observation Value")
                plt.title(f"Online Denoising Example {i+1}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            break  # Show only one batch for visual inspection.

if __name__ == "__main__":
    main()
# Note: The above code assumes that the clean and noisy datasets are saved as .npy files.