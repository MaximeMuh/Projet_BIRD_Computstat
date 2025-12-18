import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


input_dim, hidden_dim, latent_dim = 784, 200, 50
batch_size, learning_rate, epochs = 10, 1e-4, 100

if torch.backends.mps.is_available():
    device = torch.device("mps");  print("Succès : Utilisation de l'accélération MPS (Mac M1/M2/M3) 🚀")
elif torch.cuda.is_available():
    device = torch.device("cuda"); print("Succès : Utilisation du GPU NVIDIA (CUDA)")
else:
    device = torch.device("cpu");  print("Attention : Utilisation du CPU (plus lent)")


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        # On code x -> h -> (mu, logvar)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu, self.fc_logvar = nn.Linear(hidden_dim, latent_dim), nn.Linear(hidden_dim, latent_dim)
        # On décode z -> h -> x_recon
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.relu, self.sigmoid = nn.ReLU(), nn.Sigmoid()

    def encode(self, x):
        # On fabrique les paramètres de q(z|x)
        h = self.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # On échantillonne z sans casser le gradient
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        # On reconstruit une proba par pixel (Bernoulli)
        h = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h))

    def forward(self, x):
        # On aplatit MNIST: (B,1,28,28) -> (B,784)
        x = x.view(-1, input_dim)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    # On fait BCE pour la reconstruction + KL(q||p) pour régulariser z
    bce = nn.functional.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


def get_dataloader():
    # On convertit en tenseur puis on binarise (Bernoulli)
    transform = transforms.Compose([transforms.ToTensor(), lambda x: (x > 0.5).float()])
    train_ds = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(model, optimizer, train_loader, epoch):
    # On fait une epoch d'entraînement classique
    model.train()
    train_loss = 0.0

    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader.dataset)
    print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")


if __name__ == "__main__":
    print("Chargement des données...")
    train_loader, test_loader = get_dataloader()
    print(epochs)

    # On instancie le VAE et Adam comme dans le papier
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Début de l'entraînement...")
    for epoch in range(1, epochs + 1):
        train(model, optimizer, train_loader, epoch)

    # On sauvegarde pour la suite (sampling, inpainting, etc.)
    torch.save(model.state_dict(), "vae_mnist_paper.pth")
    print("Modèle sauvegardé sous 'vae_mnist_paper.pth' !")
    print("Entraînement terminé.")