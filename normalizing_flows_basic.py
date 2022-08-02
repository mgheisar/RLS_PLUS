from __future__ import division
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import Generator
from sklearn.decomposition import IncrementalPCA
import pandas as pd
import scipy.stats as stats

torch.cuda.set_device(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(1)
torch.manual_seed(1)

dim = 4
n_samples = 10000
base_mu, base_cov = torch.zeros(dim), torch.eye(dim)
base_dist = MultivariateNormal(base_mu, base_cov)
Z = base_dist.rsample(sample_shape=(n_samples,))
# plt.scatter(Z[:, 0], Z[:, 1])
# plt.show()


class R_NVP(nn.Module):
    def __init__(self, d, k, hidden):
        super().__init__()
        self.d, self.k = d, k
        self.sig_net = nn.Sequential(
            nn.Linear(k, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, d - k))

        self.mu_net = nn.Sequential(
            nn.Linear(k, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, d - k))

    def forward(self, x, flip=False):
        x1, x2 = x[:, :self.k], x[:, self.k:]

        if flip:
            x2, x1 = x1, x2

        # forward
        sig = self.sig_net(x1)
        aa = x2 * torch.exp(sig)
        bb = self.mu_net(x1)
        z1, z2 = x1, x2 * torch.exp(sig) + self.mu_net(x1)

        if flip:
            z2, z1 = z1, z2

        z_hat = torch.cat([z1, z2], dim=-1)

        log_pz = base_dist.log_prob(z_hat)
        log_jacob = sig.sum(-1)

        return z_hat, log_pz, log_jacob

    def inverse(self, Z, flip=False):
        z1, z2 = Z[:, :self.k], Z[:, self.k:]

        if flip:
            z2, z1 = z1, z2

        x1 = z1
        x2 = (z2 - self.mu_net(z1)) * torch.exp(-self.sig_net(z1))

        if flip:
            x2, x1 = x1, x2
        return torch.cat([x1, x2], -1)


class stacked_NVP(nn.Module):
    def __init__(self, d, k, hidden, n):
        super().__init__()
        self.bijectors = nn.ModuleList([
            R_NVP(d, k, hidden=hidden) for _ in range(n)
        ])
        self.flips = [True if i % 2 else False for i in range(n)]

    def forward(self, x):
        log_jacobs = []

        for bijector, f in zip(self.bijectors, self.flips):
            x, log_pz, lj = bijector(x, flip=f)
            log_jacobs.append(lj)

        return x, log_pz, sum(log_jacobs)

    def inverse(self, z):
        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):
            z = bijector.inverse(z, flip=f)
        return z


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def train(loader, model, epochs, optim, scheduler):
    loader = sample_data(loader)
    losses = []
    for _ in range(epochs):
        # get batch
        X = next(loader)

        z, log_pz, log_jacob = model(X[0])
        loss = (-log_pz - log_jacob).mean()
        losses.append(loss)

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
    return losses


def view(X, model, losses):
    # plt.plot(losses)
    # plt.title("Model Loss vs Epoch")
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.show()

    # X_hat = model.inverse(Z).detach().numpy()
    # plt.scatter(X_hat[:, 0], X_hat[:, 1])
    # plt.title("Inverse of Normal Samples Z: X = F^-1(Z)")
    # plt.show()

    z, _, _ = model(torch.from_numpy(X).float())
    z = z.detach().numpy()
    plt.scatter(z[:, 0], z[:, 1])
    plt.title("Transformation of Data Samples X: Z = F(W)")
    plt.show()

    x1 = z[:, 0]
    x2 = z[:, 1]
    df = pd.DataFrame({"x": x1, "y": x2})
    pd.plotting.scatter_matrix(df, diagonal='kde')
    stat, p = stats.shapiro(x1[:5000])
    if np.min(p) > 0.05:
        print('Probability Gaussian')
    else:
        print('Probability not Gaussian')
    plt.show()


size = 1024
ckpt = "checkpoint/stylegan2-ffhq-config-f.pt"
# ckpt = "checkpoint/BBBC021/150000.pt"
# ckpt = "checkpoint/face256.pt"
g_ema = Generator(size, 512, 8).to(device)
map_location = lambda storage, loc: storage.cuda()
gan = torch.load(ckpt, map_location=map_location)
g_ema.load_state_dict(gan["g_ema"], strict=False)
g_ema.eval()
del gan
torch.cuda.empty_cache()
# lrelu = torch.nn.LeakyReLU(negative_slope=0.2)
with torch.no_grad():
    z_samples = torch.randn((2000000, 512), dtype=torch.float32, device=device)
    w_samples = g_ema.style(z_samples)
    torch.save(w_samples, "/projects/superres/Marzieh/pytorch-flows/data/w_samples_train_face1024.pt")
    # z_samples = torch.randn((5000, 512), dtype=torch.float32, device=device)
    # w_samples = g_ema.style(z_samples)
    # torch.save(w_samples, "/projects/superres/Marzieh/pytorch-flows/data/w_samples_test_face1024.pt")
    exit(0)
p_samples = torch.nn.LeakyReLU(5)(w_samples)
# psamples = p_samples.cpu().numpy()
#
# n_components = 512
# transformer = IncrementalPCA(n_components, whiten=False, batch_size=max(100, 5 * n_components))
# X_mean = psamples.mean(0)
# transformer.fit(psamples - psamples.mean(0))
# X_comp = transformer.components_
# X_stdev = np.sqrt(transformer.explained_variance_)  # already sorted
# # var_ratio = transformer.explained_variance_ratio_
# X_mean = torch.from_numpy(X_mean).float().to(device)
# X_comp = torch.from_numpy(X_comp).float().to(device)
# X_stdev = torch.from_numpy(X_stdev).float().to(device)
# pn_samples = torch.mm((p_samples - X_mean), X_comp.T) / X_stdev
subset = {"z_space": z_samples, "w_space": w_samples, "p_space": p_samples}
space = 'p_space'

measurements = subset[space].cpu().numpy()
measurements = measurements[:, :dim]
# noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
# measurements, y = noisy_moons
measurements = StandardScaler().fit_transform(measurements)
# normalize
x1 = measurements[:, 0]
x2 = measurements[:, 1]
df = pd.DataFrame({"x": x1, "y": x2})
pd.plotting.scatter_matrix(df, diagonal='kde')
stat, p = stats.shapiro(x1[:5000])
if np.min(p) > 0.05:
    print('Probability Gaussian')
else:
    print('Probability not Gaussian')
plt.show()

X = torch.from_numpy(measurements).float()
dataset = TensorDataset(X)
loader = DataLoader(dataset, batch_size=512)
d = dim
k = dim//2
# 5 Layer R_NVP

model = stacked_NVP(d, k, hidden=512, n=4)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

# training loop
losses = train(loader, model, 2000, optim, scheduler)
view(measurements, model, losses)
