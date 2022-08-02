import argparse
import torch
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from model import Generator


# "checkpoint/BBBC021/150000.pt"
# "checkpoint/BlueBubbleDMSO/400000.pt"

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description="GANspace vs Closed Form Factorization")
    parser.add_argument("ckpt", type=str, help="name of the model checkpoint")
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )
    args = parser.parse_args()

    map_location = lambda storage, loc: storage.cuda()
    ckpt = torch.load(args.ckpt, map_location=map_location)

    modulate = {
        k: v
        for k, v in ckpt["g_ema"].items()
        if "modulation" in k and "to_rgbs" not in k and "weight" in k
    }

    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0).to("cpu")
    # eigvec = torch.svd(W).V.to("cpu")
    pca = PCA(n_components=10)
    pca.fit(W)
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca) # cumulative sum of variance explained with [n] features
    # Plot the explained variance against cumulative explained variance
    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
             label='Cumulative explained variance')
    plt.ylabel("Explained variance ratio")
    plt.xlabel(f"Principal component index (Closed form factorization)")
    plt.title('BBBC021')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"Variance_ratio_BBBC021_Closedform.png")
    plt.show()

    # ## ----------------------------------------
    g = Generator(128, 512, 8, channel_multiplier=2).to(args.device)
    g.load_state_dict(ckpt["g_ema"], strict=False)
    latent = torch.randn(100000, 512, device=args.device)
    latent = g.get_latent(latent)
    latent = latent.to("cpu")

    pca = PCA(n_components=10)
    pca.fit(latent)
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)  # cumulative sum of variance explained with [n] features
    # Plot the explained variance against cumulative explained variance
    plt.figure()
    plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
             label='Cumulative explained variance')
    plt.ylabel("Explained variance ratio")
    plt.xlabel(f"Principal component index (GAN space)")
    plt.title('BBBC021')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"Variance_ratio_BBBC021_GANSPACE.png")
    plt.show()