import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import save_image

# Generator (ResNet-based architecture as described in the CycleGAN paper)
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
            
            # Downsampling
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Residual Blocks
            *[ResnetBlock(256) for _ in range(num_res_blocks)],

            # Upsampling
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

# Discriminator (PatchGAN-based architecture)
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

# Hyperparameters
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
image_size = 256
batch_size = 1
num_epochs = 200

# Datasets and Dataloaders
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Replace 'path_to_dataset' with your dataset path
dataset_A = datasets.ImageFolder('monet2photo/trainA', transform=transform)
dataset_B = datasets.ImageFolder('monet2photo/trainB', transform=transform)
loader_A = DataLoader(dataset_A, batch_size=batch_size, shuffle=True)
loader_B = DataLoader(dataset_B, batch_size=batch_size, shuffle=True)

# Initialize models
generator_A2B = Generator(3, 3).cuda()
generator_B2A = Generator(3, 3).cuda()
discriminator_A = Discriminator(3).cuda()
discriminator_B = Discriminator(3).cuda()

# Losses
adversarial_loss = nn.MSELoss()
cycle_loss = nn.L1Loss()
identity_loss = nn.L1Loss()

# Optimizers
optimizer_G = optim.Adam(
    list(generator_A2B.parameters()) + list(generator_B2A.parameters()), lr=lr, betas=(beta1, beta2)
)
optimizer_D_A = optim.Adam(discriminator_A.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D_B = optim.Adam(discriminator_B.parameters(), lr=lr, betas=(beta1, beta2))

# Training Loop
for epoch in range(num_epochs):
    for i, (real_A, real_B) in enumerate(zip(loader_A, loader_B)):
        real_A = real_A[0].cuda()
        real_B = real_B[0].cuda()

        # Train Generators
        optimizer_G.zero_grad()

        # Identity loss
        identity_A = generator_B2A(real_A)
        identity_B = generator_A2B(real_B)
        loss_identity_A = identity_loss(identity_A, real_A) * 5.0
        loss_identity_B = identity_loss(identity_B, real_B) * 5.0

        # GAN loss
        fake_B = generator_A2B(real_A)
        pred_fake_B = discriminator_B(fake_B)
        loss_GAN_A2B = adversarial_loss(pred_fake_B, torch.ones_like(pred_fake_B).cuda())

        fake_A = generator_B2A(real_B)
        pred_fake_A = discriminator_A(fake_A)
        loss_GAN_B2A = adversarial_loss(pred_fake_A, torch.ones_like(pred_fake_A).cuda())

        # Cycle-consistency loss
        reconstructed_A = generator_B2A(fake_B)
        reconstructed_B = generator_A2B(fake_A)
        loss_cycle_A = cycle_loss(reconstructed_A, real_A) * 10.0
        loss_cycle_B = cycle_loss(reconstructed_B, real_B) * 10.0

        # Total generator loss
        loss_G = (
            loss_identity_A
            + loss_identity_B
            + loss_GAN_A2B
            + loss_GAN_B2A
            + loss_cycle_A
            + loss_cycle_B
        )
        loss_G.backward()
        optimizer_G.step()

        # Train Discriminators
        optimizer_D_A.zero_grad()
        pred_real_A = discriminator_A(real_A)
        loss_D_real_A = adversarial_loss(pred_real_A, torch.ones_like(pred_real_A).cuda())

        pred_fake_A = discriminator_A(fake_A.detach())
        loss_D_fake_A = adversarial_loss(pred_fake_A, torch.zeros_like(pred_fake_A).cuda())

        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()

        optimizer_D_B.zero_grad()
        pred_real_B = discriminator_B(real_B)
        loss_D_real_B = adversarial_loss(pred_real_B, torch.ones_like(pred_real_B).cuda())

        pred_fake_B = discriminator_B(fake_B.detach())
        loss_D_fake_B = adversarial_loss(pred_fake_B, torch.zeros_like(pred_fake_B).cuda())

        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Step [{i}/{len(loader_A)}] \
                  Loss_G: {loss_G.item():.4f} Loss_D_A: {loss_D_A.item():.4f} Loss_D_B: {loss_D_B.item():.4f}")

    # Save images
    if epoch % 10 == 0:
        save_image(fake_A, f"output/fake_A_{epoch}.png", normalize=True)
        save_image(fake_B, f"output/fake_B_{epoch}.png", normalize=True)

print("Training completed!")
