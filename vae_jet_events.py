import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define the dataset class
class JetEventsDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = np.load(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        event = self.data[idx]
        if self.transform:
            event = self.transform(event)
        return event

# Define the VAE architecture
class VAE(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 15 * 15, 512),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128 * 15 * 15),
            nn.ReLU(),
            nn.Unflatten(1, (128, 15, 15)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_shape[0], kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Define the loss function
def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Load the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    # You can add more transformations as needed
])
dataset = JetEventsDataset('quark_gluon_jet_events.npy', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the VAE
latent_dim = 20
input_shape = (3, 125, 125)  # Channels, Height, Width
vae = VAE(input_shape, latent_dim)

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

# Define optimizer
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Training the VAE
num_epochs = 10
for epoch in range(num_epochs):
    vae.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

# Visualize original and reconstructed events
vae.eval()
with torch.no_grad():
    data = next(iter(dataloader))
    data = data.to(device)
    recon_batch, _, _ = vae(data)

    # Plot original and reconstructed events
    plt.figure(figsize=(10, 5))
    for i in range(5):  # Plot 5 samples
        plt.subplot(2, 5, i + 1)
        plt.imshow(np.transpose(data[i].cpu().numpy(), (1, 2, 0)))
        plt.title('Original')
        plt.axis('off')
        plt.subplot(2, 5, i + 6)
        plt.imshow(np.transpose(recon_batch[i].cpu().numpy(), (1, 2, 0)))
        plt.title('Reconstructed')
        plt.axis('off')
    plt.show()
