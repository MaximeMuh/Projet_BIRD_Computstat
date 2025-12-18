import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Mode Turbo : Mac MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Mode Turbo : GPU NVIDIA (CUDA)")
else:
    device = torch.device("cpu")
    print("Mode Standard : CPU")


class DeepVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=(500, 250, 100), latent_dim=50):
        super().__init__()

        h0, h1, h2 = hidden_dims

        # On code x -> h (pyramide inversée)
        self.encoder_layers = nn.Sequential(
            nn.Linear(input_dim, h0), nn.ReLU(), nn.BatchNorm1d(h0),
            nn.Linear(h0, h1),        nn.ReLU(), nn.BatchNorm1d(h1),
            nn.Linear(h1, h2),        nn.ReLU(), nn.BatchNorm1d(h2),
        )
        # On sort mu et logvar pour q(z|x)
        self.fc_mu, self.fc_logvar = nn.Linear(h2, latent_dim), nn.Linear(h2, latent_dim)

        # On décode z -> x (pyramide)
        self.decoder_input = nn.Linear(latent_dim, h2)
        self.decoder_layers = nn.Sequential(
            nn.ReLU(), nn.BatchNorm1d(h2),
            nn.Linear(h2, h1), nn.ReLU(), nn.BatchNorm1d(h1),
            nn.Linear(h1, h0), nn.ReLU(), nn.BatchNorm1d(h0),
            nn.Linear(h0, input_dim), nn.Sigmoid(),
        )

    def encode(self, x):
        # On fabrique mu, logvar à partir de x
        h = self.encoder_layers(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # On échantillonne z avec le trick : z = mu + sigma * eps
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        # On reconstruit une proba par pixel
        h = self.decoder_input(z)
        return self.decoder_layers(h)

    def forward(self, x):
        # On aplatit MNIST (B,1,28,28) -> (B,784)
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    # On prend BCE pour la reconstruction (Bernoulli) + KL(q||p) pour régulariser
    bce = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld


def get_dataloaders(batch_size=128):
    # On binarise MNIST pour coller au modèle Bernoulli
    transform = transforms.Compose([transforms.ToTensor(), lambda x: (x > 0.5).float()])

    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def train_model(model, optimizer, epochs, train_loader):
    # On entraîne, et on affiche une stat simple de temps en temps
    model.train()
    print(f"Début entraînement : {epochs} epochs")

    for epoch in range(1, epochs + 1):
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
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Average Loss (ELBO) : {avg_loss:.4f}")


if __name__ == "__main__":
    # On fixe nos hyperparams
    BATCH_SIZE, EPOCHS, LEARNING_RATE = 128, 200, 1e-3
    SAVE_PATH = "deep_vae_mnist.pth"

    # On charge les données
    train_loader, _ = get_dataloaders(BATCH_SIZE)

    # On crée le modèle et l’optimiseur
    model = DeepVAE(latent_dim=50).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # On entraîne puis on sauvegarde
    train_model(model, optimizer, EPOCHS, train_loader)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Modèle sauvegardé : {SAVE_PATH}")