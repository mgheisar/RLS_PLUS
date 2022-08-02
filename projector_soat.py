import argparse
import math
import os
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import pickle
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import lpips
from model_soat import Generator
from op import fused_leaky_relu
import numpy as np
from util_soat import *
import torchvision
import time

gpu_num = 0
torch.cuda.set_device(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def gaussian_loss(v):
    # [B, 9088]
    loss = (v - gt_mean) @ gt_cov_inv @ (v - gt_mean).transpose(1, 0)
    return loss.mean()


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                    loss
                    + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                    + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8: break
            noise = noise.reshape([1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .permute(0, 2, 3, 1)
            .to('cpu')
            .numpy()
    )


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--lr_rampup', type=float, default=0.05)
    parser.add_argument('--lr_rampdown', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=2)  # 5
    parser.add_argument('--perc', type=float, default=0.1)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--noise_ramp', type=float, default=0.75)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--noise_regularize', type=float, default=1e5)
    parser.add_argument('--n_mean_latent', type=int, default=10000)
    parser.add_argument('files', metavar='FILES', nargs='+')
    parser.add_argument('--lr_schedule', type=str, default='linear1cycledrop',
                        help='fixed, linear1cycledrop, linear1cycle')
    parser.add_argument('--save_intermediate', action="store_false", help='Whether to store and save intermediate images')
    parser.add_argument(
        "--ckpt", type=str, default="checkpoint/BBBC021/150000.pt", help="path to the model checkpoint"
    )

    args = parser.parse_args()
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True

    out_dir = './inversion_codes'
    os.makedirs(out_dir, exist_ok=True)

    n_mean_latent = args.n_mean_latent
    resize = args.size

    transform = transforms.Compose(
        [
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []

    for imgfile in args.files:
        img = transform(Image.open(imgfile).convert('RGB'))
        imgs.append(img)
        img_ar = make_image(img.unsqueeze(0))
        pil_img = Image.fromarray(img_ar[0])
        pil_img.save('input.png')

    imgs = torch.stack(imgs, 0).to(device)

    g_ema = Generator(args.size, 512, 8).to(device)
    map_location = lambda storage, loc: storage.cuda()
    g_ema.load_state_dict(torch.load(args.ckpt, map_location=map_location)['g_ema'], strict=False)
    g_ema = g_ema.eval()
    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda"), gpu_ids=[int(gpu_num)]
    )

    with torch.no_grad():
        latent_mean = g_ema.mean_latent(50000)
        latent_in = list2style(latent_mean).repeat(imgs.shape[0], 1)

    # get gaussian stats
    if not os.path.isfile('inversion_stats.npz'):
        with torch.no_grad():
            source = list2style(g_ema.get_latent(torch.randn([10000, 512]).to(device))).cpu().numpy()
            gt_mean = source.mean(0)
            gt_cov = np.cov(source, rowvar=False)

        # We show that style space follows gaussian distribution
        # An extension from this work https://arxiv.org/abs/2009.06529
        np.savez('inversion_stats.npz', mean=gt_mean, cov=gt_cov)

    data = np.load('inversion_stats.npz')
    gt_mean = torch.tensor(data['mean']).to(device).view(1, -1).float()
    gt_cov_inv = torch.tensor(data['cov']).to(device)

    # Only take diagonals
    mask = torch.eye(*gt_cov_inv.size()).to(device)
    gt_cov_inv = torch.inverse(gt_cov_inv * mask).float()
    # percept = lpips.PerceptualLoss(net='vgg', spatial=True).to(device)
    latent_in.requires_grad = True

    # optimizer = optim.Adam([latent_in], lr=args.lr, betas=(0.9, 0.999)) ###--------------------------------------
    optimizer = optim.Adam([latent_in], lr=args.lr)
    schedule_dict = {
        'fixed': lambda x: 1,
        'linear1cycle': lambda x: (9 * (1 - np.abs(x / args.steps - 1 / 2) * 2) + 1) / 10,
        'linear1cycledrop': lambda x: (9 * (
                1 - np.abs(
            x / (0.9 * args.steps) - 1 / 2) * 2) + 1) / 10 if x < 0.9 * args.steps else 1 / 10 + (
                x - 0.9 * args.steps) / (0.1 * args.steps) * (1 / 1000 - 1 / 10),
    }
    schedule_func = schedule_dict[args.lr_schedule]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_func)
    pbar = tqdm(range(args.steps))
    latent_path = []
    start_t = time.time()
    min_loss = np.inf

    for i in pbar:
        t = i / args.steps
        # lr = get_lr(t, args.lr, rampdown=args.lr_rampdown)
        optimizer.zero_grad()
        latent_n = latent_in

        img_gen, _ = g_ema(style2list(latent_n))

        batch, channel, height, width = img_gen.shape

        # if height > 128:
        #     img_gen = F.interpolate(img_gen, size=(128, 128), mode='area')

        p_loss = percept(img_gen, imgs).mean()
        mse_loss = F.l1_loss(img_gen, imgs)
        g_loss = gaussian_loss(latent_n)

        loss = args.perc * p_loss + 1 * mse_loss
        if loss < min_loss:
            min_loss = loss
            best_summary = f'Percept: {p_loss.item():.4f};L1: {mse_loss.item():.4f}; ' \
                           f'Noise_loss: {g_loss.item():.4f};TOTAL: {loss:.4f},'
            best_im = img_gen.clone().detach()
            best_latent = latent_n.clone().detach()
            if args.save_intermediate:
                img_ar = make_image(best_im)
                for j, input_name in enumerate(args.files):
                    img_name = "inversion_codes/" + os.path.splitext(os.path.basename(input_name))[
                        0] + "-project-%d.png" % i
                    pil_img = Image.fromarray(img_ar[j])
                    pil_img.save(img_name)

        loss.backward()
        optimizer.step()
        scheduler.step()
        if (i + 1) % 1 == 0:  # 100
            latent_path.append(latent_in.detach().clone())

        pbar.set_description(
            (
                f'{i}:percept:{p_loss.item():.4f};'
                f' L1:{mse_loss.item():.4f};g: {g_loss.item():.4f};lr:{optimizer.param_groups[0]["lr"]:.4f}'
            )
        )
    total_t = time.time() - start_t
    print(f'time: {total_t:.1f}')
    result_file = {}

    latent_path.append(latent_in.detach().clone())
    # img_gen, _ = g_ema(style2list(latent_path[-1]))
    img_gen = best_im
    filename = f'{out_dir}/{os.path.splitext(os.path.basename(args.files[0]))[0]}.pt'

    img_ar = make_image(img_gen)

    for i, input_name in enumerate(args.files):
        result_file['latent'] = best_latent[i]
        img_name = os.path.splitext(os.path.basename(input_name))[
                       0] + "-project_p" + str(args.perc) + "lr" + str(args.lr) + ".png"
        pil_img = Image.fromarray(img_ar[i])
        pil_img.save(img_name)
    print(f'lr={args.lr}, percep={args.perc}')
    print(best_summary)
# torch.save(result_file, filename)
