import torch
import math
import numpy as np


def set_seed(seed=0):
    """Sets deterministic behavior and seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_generator(ckpt_path, size):
    """
    Loads a generator model from checkpoint and sets it to evaluation mode.

    Args:
        ckpt_path (str): Path to the checkpoint.
        size (int): Output image size for the generator.

    Returns:
        Generator: The loaded generator.
    """
    from model import Generator  # Import locally to avoid circular dependencies.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g_ema = Generator(size, 512, 8).to(device)
    state = torch.load(ckpt_path, map_location=device)
    g_ema.load_state_dict(state["g_ema"], strict=False)
    g_ema.eval()
    return g_ema


def create_noises(generator, num_trainable_layers=0):
    """
    Creates noise tensors for the generator.

    Args:
        generator: The generator model (used to access n_latent).
        num_trainable_layers (int): How many noise tensors should require gradients.

    Returns:
        tuple: A list of all noise tensors and a list of those with gradients enabled.
    """
    device = next(generator.parameters()).device
    noises = []
    noise_vars = []
    for i in range(generator.n_latent - 1):
        res = (1, 1, 2 ** ((i + 1) // 2 + 2), 2 ** ((i + 1) // 2 + 2))
        new_noise = torch.randn(res, dtype=torch.float, device=device)
        if i < num_trainable_layers:
            new_noise.requires_grad = True
            noise_vars.append(new_noise)
        else:
            new_noise.requires_grad = False
        noises.append(new_noise)
    return noises, noise_vars
