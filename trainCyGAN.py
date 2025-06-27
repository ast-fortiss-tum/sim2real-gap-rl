#!/usr/bin/env python3
"""
A minimal PyTorch CycleGAN example script for the monet2photo dataset,
including:
 - saving model weights
 - printing the number of parameters
 - using TensorBoard to log losses and sample images
 - setting a random seed for reproducibility
 - seeding DataLoader for deterministic shuffling

Author: Cristian Cubides (with TensorBoard + param counting added)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter  # TensorBoard

import random
import numpy as np
from torchvision.transforms.functional import InterpolationMode


# -------------------------------
# 1. Model Architectures
# -------------------------------

def step_decay(epoch, initial_lr=1.0, decay_factor=0.5):
    """
    Halve the learning rate every epoch.

    Args:
        epoch (int): Current epoch number.
        initial_lr (float): Initial learning rate scaling factor.
        decay_factor (float): Factor by which to decay the LR.

    Returns:
        float: Scaling factor for the current epoch.
    """
    return initial_lr * (decay_factor ** epoch)


class ResidualBlock(nn.Module):
    """A simple residual block for the generator."""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    CycleGAN Generator:
    Uses downsampling, residual blocks, and upsampling.
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_residual_blocks=6):
        super(Generator, self).__init__()
        # Initial convolution block
        model = [
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        # Downsampling (2 times)
        in_channels = ngf
        out_channels = ngf * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
            out_channels *= 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_channels)]

        # Upsampling (2 times)
        out_channels = in_channels // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
            in_channels = out_channels
            out_channels //= 2

        # Output layer
        model += [
            nn.Conv2d(in_channels, output_nc, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """
    CycleGAN Discriminator (PatchGAN).
    """
    def __init__(self, input_nc=3, ndf=64):
        super(Discriminator, self).__init__()

        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        in_channels = ndf
        out_channels = ndf * 2
        model += [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        in_channels = out_channels
        out_channels = ndf * 4
        model += [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # The final output of PatchGAN is a feature map (not a single scalar).
        model += [
            nn.Conv2d(out_channels, 1, kernel_size=4, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# -------------------------------
# 2. CycleGAN Losses
# -------------------------------
def gan_loss(prediction, target_is_real):
    """
    Computes standard GAN loss (MSE).
    `target_is_real` is True if real, False if fake.
    """
    if target_is_real:
        target_tensor = torch.ones_like(prediction)
    else:
        target_tensor = torch.zeros_like(prediction)
    return nn.MSELoss()(prediction, target_tensor)


def cycle_consistency_loss(recovered, real):
    """
    L1 cycle consistency loss: encourages G_AB(G_BA(x)) ~ x.
    """
    return nn.L1Loss()(recovered, real)


def identity_loss(generated, real):
    """
    L1 identity loss: G_A->B(B) should be close to B if B used as input, etc.
    """
    return nn.L1Loss()(generated, real)


# -------------------------------
# 3. Helper: Count parameters
# -------------------------------
def count_params(model):
    """
    Counts the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -------------------------------
# 4. Main Training Function
# -------------------------------
def train_cyclegan(
    data_root="sim2Car_complete",
    epochs=100,
    batch_size=1,
    lr=0.0005,  # Initial LR set to 0.001
    device="cpu",
    save_dir="checkpoints_Car",
    log_dir="runs/cyclegan_car",
    seed=None
):
    # -----------------------
    # 1. Set Random Seed
    # -----------------------
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        print(f'No seed provided. Using randomly generated seed: {seed}')
    else:
        print(f'Setting random seed to: {seed}')

    # Set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Optionally, set deterministic flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # -----------------------
    # 2. Transforms & Datasets
    # -----------------------
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # normalize to [-1, 1]
    ])

    datasetA = datasets.ImageFolder(root=data_root, transform=transform)
    datasetA.samples = [
        (path, 0) for (path, class_idx) in datasetA.samples
        if "trainA" in path  # only Monet images
    ]

    datasetB = datasets.ImageFolder(root=data_root, transform=transform)
    datasetB.samples = [
        (path, 0) for (path, class_idx) in datasetB.samples
        if "trainB" in path  # only Photo images
    ]

    # -----------------------
    # 3. DataLoader Seed Setup
    # -----------------------
    # Create a generator for DataLoader shuffling
    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(seed)

    # Define worker_init_fn to ensure each worker has a deterministic seed
    def worker_init_fn(worker_id):
        # Set the seed for each worker based on the global seed and worker ID
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # Create DataLoaders with the generator and worker_init_fn
    loaderA = DataLoader(
        datasetA,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=data_loader_generator,  # Pass the generator here
        num_workers=4,  # Set to 0 for maximum reproducibility
        worker_init_fn=worker_init_fn  # Initialize workers with the seed
    )

    loaderB = DataLoader(
        datasetB,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=data_loader_generator,  # Pass the generator here
        num_workers=4,  # Set to 0 for maximum reproducibility
        worker_init_fn=worker_init_fn  # Initialize workers with the seed
    )

    print(f"Loader A size: {len(loaderA)} batches")
    print(f"Loader B size: {len(loaderB)} batches")
    print(f"Data root: {data_root}")

    # -----------------------
    # 4. Model Init
    # -----------------------
    netG_AB = Generator().to(device)  # Monet -> Photo
    netG_BA = Generator().to(device)  # Photo -> Monet
    netD_A  = Discriminator().to(device)  # Real/Fake Monet
    netD_B  = Discriminator().to(device)  # Real/Fake Photo

    # Print number of parameters for each model
    print(f"Number of params in netG_AB (Monet->Photo): {count_params(netG_AB):,}")
    print(f"Number of params in netG_BA (Photo->Monet): {count_params(netG_BA):,}")
    print(f"Number of params in netD_A (Monet Discriminator): {count_params(netD_A):,}")
    print(f"Number of params in netD_B (Photo Discriminator): {count_params(netD_B):,}")

    # -----------------------
    # 5. TensorBoard Writer
    # -----------------------
    writer = SummaryWriter(log_dir=log_dir)

    # -----------------------
    # 6. Optimizers
    # -----------------------
    g_params = list(netG_AB.parameters()) + list(netG_BA.parameters())
    optimizer_G = optim.Adam(g_params, lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

    # -----------------------
    # 7. Schedulers
    # -----------------------
    # Implement stepwise decay: halve the LR every epoch
    scheduler_G = LambdaLR(
        optimizer_G,
        lr_lambda=lambda epoch: step_decay(epoch, initial_lr=1.0, decay_factor=0.5)
    )
    scheduler_D_A = LambdaLR(
        optimizer_D_A,
        lr_lambda=lambda epoch: step_decay(epoch, initial_lr=1.0, decay_factor=0.5)
    )
    scheduler_D_B = LambdaLR(
        optimizer_D_B,
        lr_lambda=lambda epoch: step_decay(epoch, initial_lr=1.0, decay_factor=0.5)
    )

    # Create checkpoint directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------
    # 8. Training Loop
    # -----------------------
    global_step = 0
    for epoch in range(epochs):

        # -----------------------
        # Print Current Learning Rate
        # -----------------------
        current_lr_G = optimizer_G.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, Current LR = {current_lr_G:.6f}")

        # -----------------------
        # Ensure both dataloaders have the same number of steps per epoch
        # -----------------------
        loader_size = min(len(loaderA), len(loaderB))
        iterA = iter(loaderA)
        iterB = iter(loaderB)

        for i in range(loader_size):
            realA, _ = next(iterA)  # Monet
            realB, _ = next(iterB)  # Photos
            realA = realA.to(device)
            realB = realB.to(device)

            # -----------------------
            #  Train Generators
            # -----------------------
            optimizer_G.zero_grad()

            # A -> B
            fakeB = netG_AB(realA)
            pred_fakeB = netD_B(fakeB)
            loss_g_ab = gan_loss(pred_fakeB, True)  # want D_B to see fakeB as real

            # B -> A
            fakeA = netG_BA(realB)
            pred_fakeA = netD_A(fakeA)
            loss_g_ba = gan_loss(pred_fakeA, True)  # want D_A to see fakeA as real

            # Cycle consistency
            recovA = netG_BA(fakeB)  # A -> B -> A
            recovB = netG_AB(fakeA)  # B -> A -> B
            loss_cycleA = cycle_consistency_loss(recovA, realA)
            loss_cycleB = cycle_consistency_loss(recovB, realB)

            # Identity loss (optional, often helps preserve color)
            idtA = netG_BA(realA)
            idtB = netG_AB(realB)
            loss_idtA = identity_loss(idtA, realA) * 0.5
            loss_idtB = identity_loss(idtB, realB) * 0.5

            # Total G loss
            loss_G = (loss_g_ab + loss_g_ba) \
                     + 10.0*(loss_cycleA + loss_cycleB) \
                     + 5.0*(loss_idtA + loss_idtB)
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A (real vs. fake Monet)
            # -----------------------
            optimizer_D_A.zero_grad()
            # Real Monet
            pred_realA = netD_A(realA)
            loss_d_realA = gan_loss(pred_realA, True)
            # Fake Monet
            pred_fakeA = netD_A(fakeA.detach())  # detach to avoid backprop to G
            loss_d_fakeA = gan_loss(pred_fakeA, False)
            loss_D_A = 0.5 * (loss_d_realA + loss_d_fakeA)
            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B (real vs. fake Photo)
            # -----------------------
            optimizer_D_B.zero_grad()
            # Real Photo
            pred_realB = netD_B(realB)
            loss_d_realB = gan_loss(pred_realB, True)
            # Fake Photo
            pred_fakeB = netD_B(fakeB.detach())
            loss_d_fakeB = gan_loss(pred_fakeB, False)
            loss_D_B = 0.5 * (loss_d_realB + loss_d_fakeB)
            loss_D_B.backward()
            optimizer_D_B.step()

            # -----------------------
            # Log to TensorBoard
            # -----------------------
            writer.add_scalar("Loss/Generator", loss_G.item(), global_step)
            writer.add_scalar("Loss/D_A",       loss_D_A.item(), global_step)
            writer.add_scalar("Loss/D_B",       loss_D_B.item(), global_step)

            # -----------------------
            # Optionally, log sample images every N steps
            # -----------------------
            if (epoch % 1 == 0):
                # Write a few images to see the transformations .  . 
                # We clamp to [0,1] because the images are [-1,1] range
                writer.add_images("Real/Sim", (realA * 0.5 + 0.5).clamp(0,1), epoch)
                writer.add_images("Real/Car", (realB * 0.5 + 0.5).clamp(0,1), epoch)
                writer.add_images("Fake/Sim", (fakeA * 0.5 + 0.5).clamp(0,1), epoch)
                writer.add_images("Fake/Car", (fakeB * 0.5 + 0.5).clamp(0,1), epoch)

            global_step += 1

            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{loader_size}] "
                      f"D_A: {loss_D_A.item():.4f}, D_B: {loss_D_B.item():.4f}, G: {loss_G.item():.4f}")

        # -----------------------
        # Step the LR schedulers **once per epoch**:
        # -----------------------
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        # -----------------------
        # Log the updated learning rates to TensorBoard
        # -----------------------
        writer.add_scalar("LR/Generator", optimizer_G.param_groups[0]['lr'], epoch)
        writer.add_scalar("LR/Discriminator_A", optimizer_D_A.param_groups[0]['lr'], epoch)
        writer.add_scalar("LR/Discriminator_B", optimizer_D_B.param_groups[0]['lr'], epoch)

        # -----------------------
        # Save checkpoints at the end of each epoch
        # -----------------------
        if epoch % 1 == 0:
            torch.save(netG_AB.state_dict(), os.path.join(save_dir, f"CarF_netG_AB_epoch_{epoch+1}.pth"))
            torch.save(netG_BA.state_dict(), os.path.join(save_dir, f"CarF_netG_BA_epoch_{epoch+1}.pth"))
            torch.save(netD_A.state_dict(), os.path.join(save_dir, f"CarF_netD_A_epoch_{epoch+1}.pth"))
            torch.save(netD_B.state_dict(), os.path.join(save_dir, f"CarF_netD_B_epoch_{epoch+1}.pth"))

    writer.close()  # close the TensorBoard writer
    print("Training complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CycleGAN Monet2Photo Example with Model Saving, Param Counting, TensorBoard Logging, Random Seed, and DataLoader Seeding"
    )
    parser.add_argument("--data_root", type=str, default="sim2Car_complete",
                        help="Path to the monet2photo dataset folder")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (1 is common in CycleGAN)")
    parser.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="runs/cyclegan_car2",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: random seed)")
    args = parser.parse_args()

    print("Using device:", args.device)
    train_cyclegan(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        seed=args.seed
    )
