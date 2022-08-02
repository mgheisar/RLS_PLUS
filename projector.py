# # implementation of image2stylegan++
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import math
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import lpips
from model import Generator
import time
import numpy as np

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True

gpu_num = 0
torch.cuda.set_device(gpu_num)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --ckpt "checkpoint/BBBC021/150000.pt" "sample_g/Week3_25681_B03_1_104_729.png"
# "checkpoint/BlueBubbleDMSO/150000.pt"
# "checkpoint/Golgi/250000.pt" "input/HR/imgt0.png"
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

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
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
            .to("cpu")
            .numpy()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--size", type=int, default=128, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.5, help="learning rate")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--steps", type=int, default=1000, help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=0,
        help="weight of the noise regularization",
    )
    parser.add_argument("--mse", type=float, default=1, help="weight of the mse loss")
    parser.add_argument(
        "--w_plus",
        action="store_false",
        help="allow to use distinct latent codes to each layers",
    )
    parser.add_argument(
        "files", metavar="FILES", nargs="+", help="path to image files to be projected"
    )
    parser.add_argument('--save_intermediate', action="store_false",
                        help='Whether to store and save intermediate images during optimization')
    parser.add_argument('--lr_schedule', type=str, default='linear1cycledrop',
                        help='fixed, linear1cycledrop, linear1cycle')

    args = parser.parse_args()

    n_mean_latent = 1000000

    # resize = min(args.size, 256)
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
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)
        img_ar = make_image(img.unsqueeze(0))
        pil_img = Image.fromarray(img_ar[0])
        pil_img.save('input.png')

    imgs = torch.stack(imgs, 0).to(device)

    g_ema = Generator(args.size, 512, 8).to(device)
    map_location = lambda storage, loc: storage.cuda()
    g_ema.load_state_dict(torch.load(args.ckpt, map_location=map_location)["g_ema"], strict=False)
    g_ema.eval()
    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda"), gpu_ids=[int(gpu_num)]
    )

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())
    for i, noise in enumerate(noises):
        noise.requires_grad = True

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    # loss_str = "1*L1+1*Percept+0.1*Adv"
    # latent_dir = "Domain_Projection_Prev/1000Steps_without_noise/demecolcine_10.0"
    # dt_list = torch.load(latent_dir + loss_str + ".pt", map_location=map_location)
    # dt_files = list(dt_list.keys())
    # dt_latent = []
    # for i in range(len(dt_files)):
    #     dt_latent.append(dt_list[dt_files[i]]['latent'].unsqueeze(0))
    # dt_latent = torch.stack(dt_latent)
    # latent_mean = torch.mean(dt_latent, dim=0)
    # latent_in = latent_mean.detach().clone()
    if args.w_plus:
        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)
        latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
    else:
        latent_in = latent_mean.detach().clone().repeat(imgs.shape[0], 1)

    latent_in.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=args.lr)
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
        # lr = get_lr(t, args.lr, args.lr_rampdown)
        # optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad()
        # noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        # latent_n = latent_noise(latent_in, noise_strength.item())
        img_gen, _ = g_ema([latent_in], input_is_latent=True, noise=noises)

        batch, channel, height, width = img_gen.shape

        # if height > 256:
        #     factor = height // 256
        #
        #     img_gen = img_gen.reshape(
        #         batch, channel, height // factor, factor, width // factor, factor
        #     )
        #     img_gen = img_gen.mean([3, 5])

        p_loss = percept(img_gen, imgs).mean()
        n_loss = noise_regularize(noises)
        # mse_loss = F.mse_loss(img_gen, imgs)
        mse_loss = F.l1_loss(img_gen, imgs)

        loss = 0.5 * p_loss + 1 * mse_loss + args.noise_regularize * n_loss
        if loss < min_loss:
            min_loss = loss
            best_summary = f'Percept: {p_loss.item():.4f};L1: {mse_loss.item():.4f}; ' \
                           f'Noise_loss: {n_loss.item():.4f};TOTAL: {loss:.4f},'
            best_im = img_gen.detach().clone()
            best_latent = latent_in.detach().clone()
            best_noises = noises
        loss.backward()
        optimizer.step()
        scheduler.step()

        noise_normalize_(noises)

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description(
            (
                f"Percept: {p_loss.item():.4f};"
                f" L1: {mse_loss.item():.4f};"
                f" Noise_loss: {n_loss.item():.4f};|"
                f" TOTAL: {loss:.4f},"

            )
        )
        if args.save_intermediate:
            img_ar = make_image(img_gen)

            for j, input_name in enumerate(args.files):
                img_name = "input/project/gif/" + os.path.splitext(os.path.basename(input_name))[0] + "-project-%d.png" % i
                pil_img = Image.fromarray(img_ar[j])
                pil_img.save(img_name)
    total_t = time.time() - start_t
    print(f'time: {total_t:.1f}')
    img_gen, _ = g_ema([best_latent], input_is_latent=True, noise=noises)
    filename = os.path.splitext(os.path.basename(args.files[0]))[0] + ".pt"
    img_ar = make_image(img_gen)
    # torch.save(best_latent, "w_i2s")
    # torch.save(noises, "w_i2s_n")

    result_file = {}
    for i, input_name in enumerate(args.files):
        noise_single = []
        for noise in noises:
            noise_single.append(noise[i: i + 1])
        result_file[input_name] = {
            "img": img_gen[i],
            "latent": latent_in[i],
            "noise": noise_single,
        }
        img_name = os.path.splitext(os.path.basename(input_name))[0] + f'i2s-project_w{args.w_plus}_n0.png'
        pil_img = Image.fromarray(img_ar[i])
        pil_img.save(f'input/project/{img_name}')
        print(best_summary)

    # torch.save(result_file, filename)
