# # implementation of PULSE with NF Gaussianization
import os
import argparse
import pickle
import torch
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import lpips
from model import Generator, Discriminator
import time
import numpy as np
from FLOWS import flows as fnn
import math
from bicubic import BicubicDownSample
import glob

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


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


def project_onto_l1_ball(x, eps):
    """
    See: https://gist.github.com/tonyduan/1329998205d88c566588e57e3e2c0c55
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


def cvt(x):
    return x.to(device=device, dtype=dtype, memory_format=torch.contiguous_format)


def cross(latent):
    if len(latent.shape) == 2:
        return 0
    DD = 0
    for i in range(latent.shape[0]):
        latent_ = latent[i, :].unsqueeze(0)
        # X = latent_.view(-1, 1, latent_.shape[1], latent_.shape[2])
        # Y = latent_.view(-1, latent_.shape[1], 1, latent_.shape[2])
        # A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
        A\
            = torch.cdist(latent_, latent_, p=2)
        D = torch.sum(torch.triu(A, diagonal=1)) / ((latent_.shape[1] - 1) * latent_.shape[1] / 2)
        DD += D

    return DD / latent.shape[0]


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


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
        "--size", type=int, default=1024, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.1,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
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
        "--files", metavar="FILES", nargs="+", help="path to image files to be projected"
    )
    parser.add_argument('--save_intermediate', action="store_true",
                        help='Whether to store and save intermediate images during optimization')
    parser.add_argument('--lr_schedule', type=str, default='linear1cycledrop',
                        help='fixed, linear1cycledrop, linear1cycle')

    # ---------------------------------------------------
    parser.add_argument("--input_dir", type=str, default="input/input", help="output directory")
    parser.add_argument('--factor', type=int, default=8, help='Super resolution factor')
    parser.add_argument("--steps", type=int, default=1000, help="optimize iterations")
    parser.add_argument("--lr", type=float, default=0.5, help="learning rate")
    parser.add_argument('--logp', type=float, default=0.005, help='logp regularization')
    parser.add_argument('--cross', type=float, default=0, help='cross regularization')
    parser.add_argument('--pnorm', type=float, default=0, help='pnorm regularization')
    parser.add_argument("--out_dir", type=str, default="", help="output directory")
    parser.add_argument("--gpu_num", type=int, default=1, help="gpu number")
    # -------NF params------------------------------------------------------------------
    parser.add_argument(
        '--arch', choices=['icnn', 'icnn2', 'icnn3', 'denseicnn2', 'resicnn2'], type=str, default='icnn2',
    )
    parser.add_argument(
        '--softplus-type', choices=['softplus', 'gaussian_softplus'], type=str, default='gaussian_softplus',
    )
    parser.add_argument(
        '--zero-softplus', type=eval, choices=[True, False], default=True,
    )
    parser.add_argument(
        '--symm_act_first', type=eval, choices=[True, False], default=False,
    )
    parser.add_argument(
        '--trainable-w0', type=eval, choices=[True, False], default=True,
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save', help='directory to save results', type=str,
                        default='/projects/superres/Marzieh/pytorch-flows/experiments/maf_face1024')
    parser.add_argument('--dimh', type=int, default=4096)  # 64:img
    parser.add_argument('--nhidden', type=int, default=5)  # 4:img
    parser.add_argument("--nblocks", type=int, default=10, help='Number of stacked CPFs.')  # 8-8-8 img
    parser.add_argument('--atol', type=float, default=1e-3)
    parser.add_argument('--rtol', type=float, default=0.0)
    parser.add_argument('--fp64', action='store_true', default=False)
    parser.add_argument('--brute_val', action='store_true', default=False)

    args = parser.parse_args()
    gpu_num = args.gpu_num
    torch.cuda.set_device(gpu_num)
    cuda = torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_mean_latent = 1000000

    transform = transforms.Compose(
        [
            # transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs, imgs_prior = [], []
    if args.files is None:
        args.files = sorted(glob.glob(os.path.join(args.input_dir, '*x.png')))
    for imgfile in args.files:
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)
        img = transform(Image.open(imgfile.split('_')[0] + '_HR.png').convert("RGB"))
        imgs_prior.append(img)
        # img_ar = make_image(img.unsqueeze(0))
        # pil_img = Image.fromarray(img_ar[0])
        # pil_img.save('input.png')

    imgs = torch.stack(imgs, 0).to(device)
    imgs_prior = torch.stack(imgs_prior, 0).to(device)

    # # # -------Loading NF model----------------------------------------------------------------------------
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
    map_location = lambda storage, loc: storage.cuda()
    best_model_path = torch.load(os.path.join(args.save, 'best_model.pth'), map_location=map_location)
    flow.load_state_dict(best_model_path['model'])
    flow.to(device)
    with open('wlatent_face1024.pkl', 'rb') as f:
        data = pickle.load(f)
    flow.eval()
    for param in flow.parameters():
        param.requires_grad = False
    # ---------------------------------------------------------------------------------------------------

    g_ema = Generator(args.size, 512, 8).to(device)
    map_location = lambda storage, loc: storage.cuda()
    g_ema.load_state_dict(torch.load(args.ckpt, map_location=map_location)["g_ema"], strict=False)
    g_ema.eval()

    # discriminator = Discriminator(args.size).to(device)
    #
    # discriminator.load_state_dict(torch.load(args.ckpt, map_location=map_location)["d"], strict=False)
    # discriminator.eval()
    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda"), gpu_ids=[int(gpu_num)]
    )

    with torch.no_grad():
        latent = torch.randn((n_mean_latent, 512), dtype=torch.float32, device=device)
        latent_out = g_ema.style(latent)
        latent_mean = latent_out.mean(0)
        gaussian_fit = {"mean": latent_out.mean(0).to(device), "std": latent_out.std(0).to(device)}
    duplicate = 0

    noises = []  # stores all of the noise tensors
    noise_vars = []  # stores the noise tensors that we want to optimize on
    # num_trainable_noise_layers = g_ema.n_latent
    num_trainable_noise_layers = 0
    for i in range(g_ema.n_latent):
        # dimension of the ith noise tensor
        res = (1, 1, 2 ** ((i + 1) // 2 + 2), 2 ** ((i + 1) // 2 + 2))
        new_noise = torch.randn(res, dtype=torch.float, device=device)
        if i < num_trainable_noise_layers:
            new_noise.requires_grad = True
            noise_vars.append(new_noise)
        else:
            new_noise.requires_grad = False
        noises.append(new_noise)
    lr_vec = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    logp_vec = [0, 0, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.0005, 0.01, 0.005, 0.001, 0, 0.005, 0.005,
                0.005, 0.005, 0.005, 0.005, 0.0005]
    cross_vec = [0.5, 0.5, 0, 0.01, 0.1, 0.1, 0.05, 0.5, 0.5, 0.05, 0, 0.1, 0.1, 0, 0.01, 0.05, 0.05, 0.1, 0.1, 0.5]
    pnorm_vec = [0.01, 0.05, 0.1, 0.1, 0.5, 0.05, 0.1, 0.05, 0.01, 0, 0.05, 0.01, 0.05, 0, 0.05, 0, 0.05, 0.01, 0, 0]
    for i_lr in range(len(lr_vec)):
        args.lr = lr_vec[i_lr]
        args.logp = logp_vec[i_lr]
        args.cross = cross_vec[i_lr]
        args.pnorm = pnorm_vec[i_lr]
        if args.w_plus:
            latent = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)
            latent_in = latent.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
        else:
            latent_in = latent_mean.detach().clone().repeat(imgs.shape[0], 1)

        # latent_mean_ = torch.load('w_nf_plus_o', map_location=map_location)
        # latent_in = latent_mean_.detach().clone()

        latent_in.requires_grad = True
        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adamax': torch.optim.Adamax
        }
        # latent = latent_in
        var_list = [latent_in] + noise_vars
        optimizer = torch.optim.Adam(var_list, lr=args.lr)
        # optimizer = SphericalOptimizer(optim.Adam, [latent_in] + noise_vars, lr=args.lr)
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
        # pbar = tqdm(range(args.steps))
        latent_path = []
        start_t = time.time()
        min_loss = np.inf

        Downsampler = BicubicDownSample(factor=args.factor, device=device, normalize=False)
        for i in range(args.steps):
            t = i / args.steps
            optimizer.zero_grad()
            img_gen, _ = g_ema([latent_in], input_is_latent=True, noise=noises)
            # #-----NF ---------------------------------------------------------------------------
            l1_loss = F.l1_loss(Downsampler(img_gen), imgs)
            mse_loss = F.mse_loss(Downsampler(img_gen), imgs)
            # latent_ = latent_in
            # n_loss = noise_regularize(noises)
            x = latent_in.view(-1, latent_in.size(-1))
            # x = torch.nn.LeakyReLU(5)(x)
            x = (x - torch.from_numpy(data["mu"]).to(x)) / torch.from_numpy(data["std"]).to(x)
            logp = flow.log_probs(x, None).mean()
            logp_loss = -logp

            a = (torch.ones(1) * math.sqrt(512)).to(device)
            x = latent_in.view(-1, latent_in.size(-1))
            x = (x - torch.from_numpy(data["mu"]).to(x)) / torch.from_numpy(data["std"]).to(x)
            x = flow.forward(x, None, mode='direct')[0]
            p_norm_loss = torch.pow(torch.mean(torch.norm(x, dim=-1)) - a, 2)

            # fake_pred = discriminator(img_gen)
            # adv_loss = F.softplus(-fake_pred).mean()
            loss = 10 * l1_loss + args.logp * logp_loss + args.pnorm * p_norm_loss
            cross_loss = 0
            if args.w_plus:
                cross_loss = cross(latent_in)
                loss += args.cross * cross_loss
            if loss < min_loss:
                min_loss = loss
                best_summary = f'L1: {l1_loss.item():.3f}; L2: {mse_loss.item():.3f}; cross: {cross_loss:.3f};' \
                              f'logp: {logp_loss: 3f}; pn: {p_norm_loss.item():.3f},'
                best_im = img_gen.detach().clone()
                best_im_LR = Downsampler(img_gen)
                best_latent = latent_in.detach().clone()
                best_noises = noises
                best_step = i + 1
                best_l1 = l1_loss
            if torch.isnan(loss):
                break
            loss.backward()
            optimizer.step()
            scheduler.step()

            # # deviation = project_onto_l1_ball(latent_in - latent_mean_, 100)
            # # var_list[0].data = latent_mean_ + deviation
            # # test = torch.norm(deviation, p=1, dim=1)
            # deviations = [project_onto_l1_ball(lat_in - lat_m, 128) for lat_in, lat_m in zip(latent_in, latent_mean_)]
            # a = torch.stack(deviations, 0)
            # var_list[0].data = latent_mean_ + torch.stack(deviations, 0)

            # noise_normalize_(noises)

            if (i + 1) % 100 == 0:
                latent_path.append(latent_in.detach().clone())

            # pbar.set_description(
            #     (
            #         f" L2: {mse_loss.item():.3f};"
            #         f" logp: {logp_loss:.3f};"
            #         f" cross: {cross_loss:.3f};"
            #         f" pn: {p_norm_loss.item():.3f},"
            #
            #     )
            # )
        if best_l1 > 0.2:
            print("Generated image might not be satisfactory. Try running the PULSE loop again.")
        else:
            total_t = time.time() - start_t
            print(f'time: {total_t:.1f}')
            # img_gen, _ = g_ema([best_latent], input_is_latent=True, noise=noises)
            img_ar = make_image(best_im)
            perceptual = percept(best_im, imgs_prior).mean()
            L1_norm = F.l1_loss(best_im, imgs_prior).mean()
            img_ar_lr = make_image(best_im_LR)
            # torch.save(best_latent, "w_nff")
            # torch.save(noises, "w_nff_n")
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

                # img_name = f'pnorm5_{duplicate}.png'
                img_name = os.path.splitext(os.path.basename(input_name))[0] + 'lr' + str(args.lr).split('.')[-1] \
                           + '_logp' + str(args.logp).split('.')[-1] + '_cross' + str(args.cross).split('.')[-1] \
                           + '_pnorm' + str(args.pnorm).split('.')[-1] + '_step' + str(best_step) + '.png'
                pil_img = Image.fromarray(img_ar[i])
                pil_img.save(f'input/project{args.out_dir}/{img_name}')
                # pil_img = Image.fromarray(img_ar_lr[i])
                # pil_img.save(f'input/project{args.out_dir}/LR_{img_name}')
                print(f'\n lr {args.lr} logp {args.logp}; cross {args.cross}; p norm {args.pnorm}')
                print(best_summary)
                print(' percept: ', perceptual.item(), 'l1:', L1_norm.item())
