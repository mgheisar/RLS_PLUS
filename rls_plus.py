import os
import argparse
import glob
import time
import math
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from bicubic import BicubicDownSample
from data_utils import Images
from rls_utils import set_seed, load_generator, create_noises


set_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def project_2_l1_ball(x, eps):
    """
    Projects tensor x onto an L1-ball with radius eps.

    Args:
        x (torch.Tensor): Input tensor.
        eps (float): Radius of the L1 ball.

    Returns:
        torch.Tensor: The projected tensor.
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)


def noise_regularize(noises):
    """
    Computes a noise regularization loss.

    Args:
        noises (list[torch.Tensor]): List of noise tensors.

    Returns:
        torch.Tensor: Regularization loss.
    """
    loss = 0
    for noise in noises:
        size = noise.shape[2]
        while True:
            loss += ((noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2) +
                     (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2))
            if size <= 8:
                break
            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2]).mean([3, 5])
            size //= 2
    return loss

def noise_normalize_(noises):
    """
    Normalizes each noise tensor in-place.

    Args:
        noises (list[torch.Tensor]): List of noise tensors.
    """
    for noise in noises:
        mean = noise.mean()
        std = noise.std()
        noise.data.add_(-mean).div_(std)


def main(args):
    torch.cuda.set_device(args.gpu_num)

    # Create output directory if it does not exist.
    if args.augs is not None:
        args.out_dir = os.path.join(args.out_dir, args.augs[0])
    os.makedirs(args.out_dir, exist_ok=True)

    # Prepare dataset and dataloader.
    image_pattern = f"{args.input_dir}/*_{args.factor}x.jpg"
    image_list = sorted(glob.glob(image_pattern))
    dataset = Images(image_list, duplicates=args.duplicate, aug=args.augs, factor=args.factor)
    dataloader = DataLoader(dataset, batch_size=args.batchsize)

    # Initialize downsampler.
    Downsampler = BicubicDownSample(factor=args.factor)

    for ref_im, ref_im_hr, ref_im_name in dataloader:
        image_id = ref_im_name[0].split("_")[0]

        # Load generator.
        g_ema = load_generator(args.ckpt, args.size)

        # Create noise tensors.
        noises, noise_vars = create_noises(g_ema, args.num_trainable_noise_layers)

        # # Compute latent_mean from a large sample.
        # with torch.no_grad():
        #     latent_samples = torch.randn((1000000, 512), dtype=torch.float32, device=device)
        #     latent_out = g_ema.style(latent_samples)
        #     latent_mean = latent_out.mean(0)

        # # Setup initial latent vector.
        # if args.duplicate == 1:
        #     # For duplicate==1, use the same latent_mean.
        #     if args.w_plus:
        #         latent = latent_mean.detach().clone().unsqueeze(0).repeat(args.batchsize, 1)
        #         latent_in = latent.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
        #     else:
        #         latent_in = latent_mean.detach().clone().repeat(args.batchsize, 1)
        # else:
        #     # For duplicates > 1, add slight random variation.
        #     if args.w_plus:
        #         latent = g_ema.style(torch.randn((args.batchsize, 512), dtype=torch.float32, device=device))
        #         latent = latent_mean + 0.2 * (latent - latent_mean)
        #         latent_in = latent.unsqueeze(1).repeat(1, g_ema.n_latent, 1).detach().clone()
        #     else:
        #         latent = g_ema.style(torch.randn((args.batchsize, 512), dtype=torch.float32, device=device))
        #         latent = latent_mean + 0.2 * (latent - latent_mean)
        #         latent_in = latent.detach().clone()

        # Load the anchor latent (w_anchor) from file.
        if not os.path.exists(args.anchor_path):
            raise FileNotFoundError(f"Anchor file not found: {anchor_path}")
        w_anchor = torch.load(anchor_path, map_location=device).to(device)

        # Reset latent_in to the anchor.
        latent_in = w_anchor.detach().clone()
        latent_in.requires_grad = True

        # Prepare optimizers.
        var_list = [latent_in] + noise_vars
        optimizer = torch.optim.Adam(var_list, lr=args.lr)
        optimizer_g = torch.optim.Adam(g_ema.parameters(), lr=0.0001)

        # Allow gradients for generator parameters.
        g_ema = g_ema.float()
        toogle_grad(g_ema, True)

        # Learning rate scheduler.
        schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9 * (1 - np.abs(x / args.steps - 0.5) * 2) + 1) / 10,
            'linear1cycledrop': lambda x: ((9 * (1 - np.abs(x / (0.9 * args.steps) - 0.5) * 2) + 1) / 10
                                           if x < 0.9 * args.steps
                                           else 1 / 10 + (x - 0.9 * args.steps) / (0.1 * args.steps) * (1 / 1000 - 1 / 10))
        }
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule_dict[args.lr_schedule])

        pbar = tqdm(range(args.steps), desc="Optimizing latent")
        start_t = time.time()
        min_loss = np.inf
        best_im = None

        for i in pbar:
            optimizer.zero_grad()
            optimizer_g.zero_grad()

            # Generate image from current latent.
            img_gen, _ = g_ema([latent_in], input_is_latent=True, noise=noises)
            img_gen = (img_gen + 1) / 2

            # Compute L1 loss between downsampled generated image and reference.
            l1_loss = F.l1_loss(Downsampler(img_gen), ref_im)
            loss = l1_loss  # + args.noise_weight * noise_regularize(noises)  # Uncomment if noise regularization is desired.

            if loss < min_loss:
                min_loss = loss
                best_im = img_gen.detach().clone()
                best_summary = f"Step {i+1}: L1 Loss = {l1_loss.item():.3f}"

            # Save the final image at the last iteration.
            if i == args.steps - 1:
                pil_img = ToPILImage()(img_gen[0].cpu().clamp(0, 1))
                final_name = f'{ref_im_name[0]}_rls_boost_last.jpg'
                pil_img.save(os.path.join(args.out_dir, final_name))

            # Optionally save intermediate images.
            if args.save_intermediate and i % 20 == 0:
                pil_img = ToPILImage()(img_gen[0].cpu().clamp(0, 1))
                intermediate_name = f'{ref_im_name[0]}_rls_boost_{i}.jpg'
                pil_img.save(os.path.join(args.out_dir, intermediate_name))

            if torch.isnan(loss):
                print("Loss is NaN. Exiting optimization loop.")
                break

            loss.backward()
            optimizer.step()
            optimizer_g.step()
            scheduler.step()

            # Normalize noise tensors.
            noise_normalize_(noises)

            # Update latent using L1-ball projection
            if args.w_plus:
                deviations = [project_2_l1_ball(lat_in - lat_m, args.radius * np.sqrt(512))
                              for lat_in, lat_m in zip(latent_in, w_anchor)]
                var_list[0].data = w_anchor + torch.stack(deviations, 0)
            else:
                deviation = project_2_l1_ball(latent_in - w_anchor, args.radius * np.sqrt(512))
                var_list[0].data = w_anchor + deviation

            pbar.set_description(f"L1 Loss: {l1_loss.item():.3f}")

        total_t = time.time() - start_t
        print(f"Total optimization time: {total_t:.1f} seconds")
        print(best_summary)

        # Save the best image.
        if best_im is not None:
            pil_img = ToPILImage()(best_im.cpu().clamp(0, 1))
            best_name = f'{ref_im_name[0]}_rls_boost_best.jpg'
            pil_img.save(os.path.join(args.out_dir, best_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimizes the latent space of a generator using an anchor latent for SR boosting."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--size", type=int, default=1024, help="Output image size of the generator")
    parser.add_argument("--input_dir", type=str, default="input/lr", help="Path to the input image directory")
    parser.add_argument("--out_dir", type=str, default="res/sr", help="Directory to save output images")
    parser.add_argument('--factor', type=int, default=16, help="Super resolution factor")
    parser.add_argument("--gpu_num", type=int, default=1, help="GPU number to use")
    parser.add_argument("--duplicate", type=int, default=1, help="Number of duplications")
    parser.add_argument('--augs', default=None, nargs='+', help="List of augmentations to test robustness",
                        choices=['rotate', 'vflip', 'hflip', 'contrast', 'brightness', 'gaussiannoise',
                                 'occlusion', 'regularblur', 'defocusblur', 'motionblur', 'gaussianblur',
                                 'saltpepper', 'perspective', 'gray', 'colorjitter'])
    parser.add_argument("--num_trainable_noise_layers", type=int, default=9, help="Number of trainable noise layers")
    parser.add_argument("--radius", type=int, default=1, help="Radius of the L1 ball for latent projection")
    parser.add_argument("--steps", type=int, default=200, help="Number of optimization iterations")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for latent and noise optimization")
    parser.add_argument("--w_plus", action="store_false", help="If set, use the W+ latent space")
    parser.add_argument("--batchsize", type=int, default=1, help="Batch size")
    parser.add_argument('--eps', type=float, default=0.5, help="Epsilon threshold for reconstruction")
    parser.add_argument('--noise_weight', type=float, default=0, help="Weight of the noise regularization term")
    parser.add_argument('--save_intermediate', action="store_true", help="Save intermediate images during optimization")
    parser.add_argument('--lr_schedule', type=str, default='linear1cycledrop',
                        choices=['fixed', 'linear1cycle', 'linear1cycledrop'], help="Learning rate schedule type")
    parser.add_argument('--anchor_path', type=str, required=True, help="Path to the anchor latent file")
    args = parser.parse_args()

    main(args)
