#!/usr/bin/env python3
"""
A minimal PyTorch CycleGAN example script for the monet2photo dataset.

Author: Cristian Cubides
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -------------------------------
# 1. Model Architectures
# -------------------------------
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
# 3. Main Training Function
# -------------------------------
def train_cyclegan(
    data_root="monet2photo",
    epochs=25,
    batch_size=1,
    lr=0.0002,
    device="cpu",
):
    # -----------------------
    # Transforms & Datasets
    # -----------------------
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # normalize to [-1, 1]
    ])

    datasetA = datasets.ImageFolder(root=data_root, transform=transform)
    datasetA.samples = [
        (path, 0) for (path, class_idx) in datasetA.samples
        if "trainA" in path  # only monet images
    ]

    datasetB = datasets.ImageFolder(root=data_root, transform=transform)
    datasetB.samples = [
        (path, 0) for (path, class_idx) in datasetB.samples
        if "trainB" in path  # only zebra images
    ]

    loaderA = DataLoader(datasetA, batch_size=batch_size, shuffle=True, drop_last=True)
    loaderB = DataLoader(datasetB, batch_size=batch_size, shuffle=True, drop_last=True)

    # -----------------------
    # Model Init
    # -----------------------
    netG_AB = Generator().to(device)  # Horse -> Monet
    netG_BA = Generator().to(device)  # Zebra -> Photo
    netD_A  = Discriminator().to(device)  # Real/Fake Monet
    netD_B  = Discriminator().to(device)  # Real/Fake Photo

    # -----------------------
    # Optimizers
    # -----------------------
    g_params = list(netG_AB.parameters()) + list(netG_BA.parameters())
    optimizer_G = optim.Adam(g_params, lr=lr, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

    # -----------------------
    # Training Loop
    # -----------------------
    for epoch in range(epochs):
        # Ensure both dataloaders have the same number of steps per epoch
        loader_size = min(len(loaderA), len(loaderB))
        iterA = iter(loaderA)
        iterB = iter(loaderB)

        for i in range(loader_size):
            realA, _ = next(iterA)  # Horses
            realB, _ = next(iterB)  # Zebras
            realA = realA.to(device)
            realB = realB.to(device)

            # -----------------------
            #  Train Generators
            # -----------------------
            optimizer_G.zero_grad()

            # A -> B
            fakeB = netG_AB(realA)
            pred_fakeB = netD_B(fakeB)
            loss_g_ab = gan_loss(pred_fakeB, True)  # want D_B to think it's real

            # B -> A
            fakeA = netG_BA(realB)
            pred_fakeA = netD_A(fakeA)
            loss_g_ba = gan_loss(pred_fakeA, True)  # want D_A to think it's real

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

            loss_G = (loss_g_ab + loss_g_ba) \
                     + 10.0*(loss_cycleA + loss_cycleB) \
                     + 5.0*(loss_idtA + loss_idtB)
            loss_G.backward()
            optimizer_G.step()

            # -----------------------
            #  Train Discriminator A (real vs. fake horses)
            # -----------------------
            optimizer_D_A.zero_grad()
            # Real horse
            pred_realA = netD_A(realA)
            loss_d_realA = gan_loss(pred_realA, True)
            # Fake horse
            pred_fakeA = netD_A(fakeA.detach())  # detach to avoid backprop to G
            loss_d_fakeA = gan_loss(pred_fakeA, False)
            loss_D_A = 0.5 * (loss_d_realA + loss_d_fakeA)
            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B (real vs. fake zebras)
            # -----------------------
            optimizer_D_B.zero_grad()
            # Real zebra
            pred_realB = netD_B(realB)
            loss_d_realB = gan_loss(pred_realB, True)
            # Fake zebra
            pred_fakeB = netD_B(fakeB.detach())
            loss_d_fakeB = gan_loss(pred_fakeB, False)
            loss_D_B = 0.5 * (loss_d_realB + loss_d_fakeB)
            loss_D_B.backward()
            optimizer_D_B.step()

            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{loader_size}] "
                      f"D_A: {loss_D_A.item():.4f}, D_B: {loss_D_B.item():.4f}, G: {loss_G.item():.4f}")

        # End of epoch

    print("Training complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CycleGAN Horse2Zebra Example")
    parser.add_argument("--data_root", type=str, default="monet2photo",
                        help="Path to the monet2photo dataset folder")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (1 is common in CycleGAN)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (e.g., 'cuda:0' or 'cpu')")
    args = parser.parse_args()

    print("Using device:", args.device)
    train_cyclegan(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device
    )
