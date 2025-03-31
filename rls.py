import os
import argparse
import glob
import time
import math
import pickle
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from FLOWS import flows as fnn
from bicubic import BicubicDownSample
from data_utils import Images
from rls_utils import set_seed, load_generator, create_noises


set_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cross(latent):
    """
    Computes a cross regularization loss for the latent codes.

    Args:
        latent (torch.Tensor): Latent tensor.

    Returns:
        torch.Tensor or float: The cross loss value.
    """
    if len(latent.shape) == 2:
        return 0
    DD = 0
    for i in range(latent.shape[0]):
        latent_ = latent[i, :].unsqueeze(0)
        A = torch.cdist(latent_, latent_, p=2)
        D = torch.sum(torch.triu(A, diagonal=1)) / (((latent_.shape[1] - 1) * latent_.shape[1]) / 2)
        DD += D
    return DD / latent.shape[0]


def load_flow(nf_path, nf_stat):
    """
    Loads the normalizing flow model and its statistics.
    """
    num_blocks = 5
    num_inputs = 512
    modules = []
    for _ in range(num_blocks):
        modules += [
            fnn.MADE(num_inputs, num_hidden=1024, num_cond_inputs=None, act='relu'),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]
    flow = fnn.FlowSequential(*modules)
    best_model_path = torch.load(os.path.join(nf_path, 'best_model.pth'), map_location=device)
    flow.load_state_dict(best_model_path['model'])
    flow.to(device)
    with open(nf_stat, 'rb') as f:
        stats = pickle.load(f)
    flow.eval()
    for param in flow.parameters():
        param.requires_grad = False
    return flow, stats



def project_onto_l1_ball(x, eps):
    """
    Projects tensor x onto the L1-ball with radius eps.
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)


def main(args):
    # Create output directory if needed.
    os.makedirs(args.out_dir, exist_ok=True)

    # Set GPU device if provided.
    torch.cuda.set_device(args.gpu_num)

    # Prepare dataset and dataloader.
    # Optionally, add a --clas argument if needed.
    image_list = sorted(glob.glob(f"{args.input_dir}/*_{args.factor}x.jpg"))[:1000]
    dataset = Images(image_list, duplicates=args.duplicate, aug=args.augs, factor=args.factor)
    dataloader = DataLoader(dataset, batch_size=args.batchsize)

    # Load Normalizing Flow and its stats.
    flow, stats = load_flow(args.nf_path, args.nf_stat)

    # Load generator.
    g_ema = load_generator(args.ckpt, args.size)

    # Create a downsampler.
    Downsampler = BicubicDownSample(factor=args.factor)

    # Create noise tensors (with zero trainable noise layers in this script).
    noises, noise_vars = create_noises(g_ema, num_trainable_layers=0)

    # Compute latent mean from many samples.
    n_mean_latent = 1000000
    with torch.no_grad():
        latent_samples = torch.randn((n_mean_latent, 512), dtype=torch.float32, device=device)
        latent_out = g_ema.style(latent_samples)
        latent_mean = latent_out.mean(0)

    # Loop over the dataset.
    for ref_im, ref_im_hr, ref_im_name in dataloader:
        image_id = ref_im_name[0].split("_")[0]

        # Initialize latent vector based on whether duplicate is 1.
        if args.duplicate == 1:
            set_seed(0)  # Ensure reproducibility.
            if args.w_plus:
                latent = latent_mean.detach().clone().unsqueeze(0).repeat(args.batchsize, 1)
                latent_in = latent.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
            else:
                latent_in = latent_mean.detach().clone().repeat(args.batchsize, 1)
        else:
            # For duplicates >1, add slight variation.
            if args.w_plus:
                latent = g_ema.style(torch.randn((args.batchsize, 512), dtype=torch.float32, device=device))
                latent = latent_mean + 0.2 * (latent - latent_mean)
                latent_in = latent.unsqueeze(1).repeat(1, g_ema.n_latent, 1).detach().clone()
            else:
                latent = g_ema.style(torch.randn((args.batchsize, 512), dtype=torch.float32, device=device))
                latent = latent_mean + 0.2 * (latent - latent_mean)
                latent_in = latent.detach().clone()

        latent_in.requires_grad = True

        # Prepare optimizer and learning rate scheduler.
        optimizer = torch.optim.Adam([latent_in] + noise_vars, lr=args.lr)
        schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9 * (1 - np.abs(x / args.steps - 0.5) * 2) + 1) / 10,
            'linear1cycledrop': lambda x: ((9 * (1 - np.abs(x / (0.9 * args.steps) - 0.5) * 2) + 1) / 10
                                           if x < 0.9 * args.steps
                                           else 1 / 10 + (x - 0.9 * args.steps) / (0.1 * args.steps) * (
                        1 / 1000 - 1 / 10))
        }
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule_dict[args.lr_schedule])

        pbar = tqdm(range(args.steps), desc="Optimizing latent")
        start_t = time.time()
        min_loss = np.inf
        best_im = None

        # Optimization loop.
        for i in pbar:
            optimizer.zero_grad()
            img_gen, _ = g_ema([latent_in], input_is_latent=True, noise=noises)
            img_gen = (img_gen + 1) / 2

            # Reconstruction loss.
            rec_loss = F.l1_loss(Downsampler(img_gen), ref_im)

            # Normalizing Flow losses.
            x = latent_in.view(-1, latent_in.size(-1))
            x = (x - torch.from_numpy(stats["mu"]).to(x)) / torch.from_numpy(stats["std"]).to(x)
            logp = flow.log_probs(x, None).mean()
            logp_loss = -logp

            a = torch.ones(1, device=device) * math.sqrt(512)
            x = latent_in.view(-1, latent_in.size(-1))
            x = (x - torch.from_numpy(stats["mu"]).to(x)) / torch.from_numpy(stats["std"]).to(x)
            x = flow.forward(x, None, mode='direct')[0]
            p_norm_loss = torch.mean((torch.norm(x, dim=-1) - a).pow(2))

            # Total loss combines L1, logp, and p-norm losses.
            loss = 10 * rec_loss + args.logp * logp_loss + args.pnorm * p_norm_loss

            # Optionally add cross regularization in W+ space.
            cross_loss = 0
            if args.w_plus:
                cross_loss = cross(latent_in)
                loss += args.cross * cross_loss

            if loss < min_loss:
                min_loss = loss
                best_im = img_gen.detach().clone()
                best_summary = (f"Step {i + 1}: L1: {rec_loss.item():.3f}; "
                                f"cross: {cross_loss:.3f}; logp: {logp_loss:.3f}; "
                                f"p_norm: {p_norm_loss.item():.3f}")

            if torch.isnan(loss):
                print("Loss is NaN. Exiting optimization loop.")
                break

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Update latent by projecting difference onto L1 ball.
            if args.w_plus:
                deviations = [project_onto_l1_ball(lat_in - lat_m, args.radius * np.sqrt(512))
                              for lat_in, lat_m in
                              zip(latent_in, latent_mean.unsqueeze(0).repeat(latent_in.shape[0], 1))]
                latent_in.data = latent_mean.unsqueeze(0).repeat(latent_in.shape[0], 1) + torch.stack(deviations, 0)
            else:
                deviation = project_onto_l1_ball(latent_in - latent_mean, args.radius * np.sqrt(512))
                latent_in.data = latent_mean + deviation

            pbar.set_description(
                f"L1: {rec_loss.item():.3f}; logp: {logp_loss:.3f}; cross: {cross_loss:.3f}; pn: {p_norm_loss.item():.3f}")

            # Save the final image at the last iteration.
            if i == args.steps - 1:
                pil_img = ToPILImage()(img_gen[0].cpu().clamp(0, 1))
                img_name = f'{ref_im_name[0]}_rls_last.jpg'
                pil_img.save(os.path.join(args.out_dir, img_name))

            # optionally save intermediate images.
            if args.save_intermediate and i % 20 == 0:
                pil_img = ToPILImage()(img_gen[0].cpu().clamp(0, 1))
                img_name = f'{ref_im_name[0]}_rls_{i}.jpg'
                pil_img.save(os.path.join(args.out_dir, img_name))

        total_t = time.time() - start_t
        print(f"Total optimization time: {total_t:.1f} seconds")
        print(best_summary)

        # Save the best image.
        best_im_LR = Downsampler(best_im)
        for j in range(args.batchsize):
            pil_img = ToPILImage()(best_im[j].cpu().clamp(0, 1))
            img_name = f'{ref_im_name[j]}_rls_best.jpg'
            pil_img.save(os.path.join(args.out_dir, img_name))

        # Optionally save the anchor latent.
        if args.save_anchor:
            anchor_dir = args.anchor_path if hasattr(args, "anchor_path") else "anchor"
            os.makedirs(anchor_dir, exist_ok=True)
            if args.duplicate == 1:
                torch.save(latent_mean, os.path.join(anchor_dir, f'latent_rls_{image_id}'))
            else:
                # For multiple duplications, save a list of best latents.
                torch.save(best_im, os.path.join(anchor_dir, f'latent_rls_{image_id}_{args.duplicate}'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent projection with NF regularization.")
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
    parser.add_argument("--save_anchor", action="store_false", help="Flag to allow saving anchor points")
    # Optimization parameters
    parser.add_argument("--steps", type=int, default=500, help="Number of optimization iterations")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    parser.add_argument('--logp', type=float, default=0.002, help="Logp regularization weight")
    parser.add_argument('--pnorm', type=float, default=0.004, help="p-norm regularization weight")
    parser.add_argument('--cross', type=float, default=0.5, help="Cross regularization weight")
    parser.add_argument("--w_plus", action="store_false", help="If set, use W+ latent space")
    parser.add_argument("--batchsize", type=int, default=1, help="Batch size")
    parser.add_argument('--eps', type=float, default=0.5, help="Epsilon threshold for reconstruction")
    # Normalizing flow parameters
    parser.add_argument('--nf_path', type=str, default='checkpoint/maf_face1024', help="Path to NF checkpoint")
    parser.add_argument("--nf_stat", type=str, default='wlatent_face1024.pkl', help="Path to latent data stats")
    # Learning rate schedule
    parser.add_argument('--lr_schedule', type=str, default='linear1cycledrop',
                        choices=['fixed', 'linear1cycle', 'linear1cycledrop'], help="Learning rate schedule type")
    # (Optional) Anchor saving path
    parser.add_argument("--anchor_path", type=str, default="anchor", help="Directory to save anchor latents")

    args = parser.parse_args()
    main(args)
