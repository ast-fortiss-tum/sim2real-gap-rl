import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import gym
import numpy as np

# Define ResNet-based Generator for RL-CycleGAN
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=9):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            *[ResnetBlock(256) for _ in range(num_res_blocks)],

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define Discriminator for RL-CycleGAN
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# Define Q-Learning Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.model(x)

# Loss functions
adversarial_loss = nn.MSELoss()
cycle_loss = nn.L1Loss()
q_consistency_loss = nn.MSELoss()

# Initialize RL-CycleGAN components
generator_A2B = Generator(3, 3).cuda()
generator_B2A = Generator(3, 3).cuda()
discriminator_A = Discriminator(3).cuda()
discriminator_B = Discriminator(3).cuda()
q_sim = QNetwork(512, 10).cuda()
q_real = QNetwork(512, 10).cuda()

# Optimizers
optimizer_G = optim.Adam(list(generator_A2B.parameters()) + list(generator_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(discriminator_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(discriminator_B.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_Q_sim = optim.Adam(q_sim.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_Q_real = optim.Adam(q_real.parameters(), lr=0.0002, betas=(0.5, 0.999))

# TensorBoard writer
writer = SummaryWriter("runs/rl_cyclegan")

# Training RL-CycleGAN
for epoch in range(100):
    for step, simulated_data in enumerate(simulated_loader):
        real_data = next(iter(real_loader))  # Assuming real_loader is smaller, loop through it cyclically
        sim_images, sim_actions = simulated_data
        real_images, real_actions = real_data

        # Sim2Real and Real2Sim translations
        fake_real = generator_A2B(sim_images.cuda())
        cycled_sim = generator_B2A(fake_real)

        fake_sim = generator_B2A(real_images.cuda())
        cycled_real = generator_A2B(fake_sim)

        # Adversarial Loss
        real_validity = discriminator_B(fake_real)
        fake_validity = discriminator_B(real_images.cuda())
        loss_GAN_A2B = adversarial_loss(real_validity, torch.ones_like(real_validity).cuda()) + adversarial_loss(fake_validity, torch.zeros_like(fake_validity).cuda())

        # Cycle Consistency Loss
        loss_cycle = cycle_loss(cycled_sim, sim_images.cuda()) + cycle_loss(cycled_real, real_images.cuda())

        # RL-Consistency Loss
        q_values_sim = q_sim(sim_images.cuda())
        q_values_real = q_real(fake_real)
        loss_q_consistency = q_consistency_loss(q_values_sim, q_values_real)

        # Total Generator Loss
        loss_G = loss_GAN_A2B + 10 * loss_cycle + 10 * loss_q_consistency
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Update Discriminators
        optimizer_D_B.zero_grad()
        real_loss = adversarial_loss(discriminator_B(real_images.cuda()), torch.ones_like(real_validity).cuda())
        fake_loss = adversarial_loss(discriminator_B(fake_real.detach()), torch.zeros_like(real_validity).cuda())
        loss_D_B = (real_loss + fake_loss) / 2
        loss_D_B.backward()
        optimizer_D_B.step()

        # Update Q Networks
        q_real_loss = q_consistency_loss(q_real(real_images.cuda()), q_sim(fake_sim.detach()))
        optimizer_Q_real.zero_grad()
        q_real_loss.backward()
        optimizer_Q_real.step()

        # Log losses to TensorBoard
        writer.add_scalar("Loss/G", loss_G.item(), epoch * len(simulated_loader) + step)
        writer.add_scalar("Loss/D_B", loss_D_B.item(), epoch * len(simulated_loader) + step)
        writer.add_scalar("Loss/Q_Consistency", loss_q_consistency.item(), epoch * len(simulated_loader) + step)

    print(f"Epoch {epoch}: Loss G: {loss_G.item()}, Loss D_B: {loss_D_B.item()}, Q Consistency Loss: {loss_q_consistency.item()}")

# Save models
torch.save(generator_A2B.state_dict(), "generator_A2B.pth")
torch.save(generator_B2A.state_dict(), "generator_B2A.pth")
torch.save(discriminator_A.state_dict(), "discriminator_A.pth")
torch.save(discriminator_B.state_dict(), "discriminator_B.pth")
torch.save(q_sim.state_dict(), "q_sim.pth")
torch.save(q_real.state_dict(), "q_real.pth")

writer.close()
