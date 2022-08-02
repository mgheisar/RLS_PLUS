# # implementation of PULSE with NF Gaussianization
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from contextlib import contextmanager
import pickle
from SphericalOptimizer import SphericalOptimizer
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
from FLOWS import flows as fnn
import math

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True

gpu_num = 0
torch.cuda.set_device(gpu_num)
cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def cvt(x):
    return x.to(device=device, dtype=dtype, memory_format=torch.contiguous_format)


def cross(latent):
    if len(latent.shape) == 2:
        return 0
    DD = 0
    for i in range(latent.shape[0]):
        latent_ = latent[i, :].unsqueeze(0)
        X = latent_.view(-1, 1, latent_.shape[1], latent_.shape[2])
        Y = latent_.view(-1, latent_.shape[1], 1, latent_.shape[2])
        A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
        D = torch.sum(torch.triu(A, diagonal=1)) / ((latent_.shape[1] - 1) * latent_.shape[1] / 2)
        DD += D
    return DD / latent.shape[0]


# def cross(latent):
#     if len(latent.shape) == 2:
#         return 0
#     DD = 0
#     for i in range(latent.shape[0]):
#         latent_ = latent[i, :].unsqueeze(0)
#         X = latent_.view(-1, 1, latent_.shape[1], latent_.shape[2])
#         Y = latent_.view(-1, latent_.shape[1], 1, latent_.shape[2])
#         A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
#         B = ((X + Y).pow(2).sum(-1) + 1e-9).sqrt()
#         D = 2 * torch.atan2(A, B)
#         D = ((D.pow(2) * latent_.shape[2]).mean((1, 2)) / 8.).sum()
#         DD += D
#     return DD / latent.shape[0]


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
        "--size", type=int, default=256, help="output image sizes of the generator"
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
    parser.add_argument('--save_intermediate', action="store_true",
                        help='Whether to store and save intermediate images during optimization')
    parser.add_argument('--lr_schedule', type=str, default='linear1cycledrop',
                        help='fixed, linear1cycledrop, linear1cycle')

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
                        default='/projects/superres/Marzieh/pytorch-flows/experiments/maf_face')
    parser.add_argument('--dimh', type=int, default=4096)  # 64:img
    parser.add_argument('--nhidden', type=int, default=5)  # 4:img
    parser.add_argument("--nblocks", type=int, default=10, help='Number of stacked CPFs.')  # 8-8-8 img
    parser.add_argument('--atol', type=float, default=1e-3)
    parser.add_argument('--rtol', type=float, default=0.0)
    parser.add_argument('--fp64', action='store_true', default=False)
    parser.add_argument('--brute_val', action='store_true', default=False)

    args = parser.parse_args()

    n_mean_latent = 1000000

    resize = min(args.size, 256)
    resize = args.size

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs = []
    # args.files = []
    # for i in range(1, 12):
    #     args.files.append(f'input/{i}.png')
    for imgfile in args.files:
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)
        img_ar = make_image(img.unsqueeze(0))
        pil_img = Image.fromarray(img_ar[0])
        pil_img.save('input.png')

    imgs = torch.stack(imgs, 0).to(device)

    # # # -------Loading NF model----------------------------------------------------------------------------
    # seed_prng(args.seed, cuda=cuda)
    # if args.fp64:
    #     torch.set_default_dtype(torch.float64)
    # dtype = torch.float32 if not args.fp64 else torch.float64
    # n_dims = 512
    # Arch = load_arch(args.arch)
    # icnns = [Arch(n_dims, args.dimh, args.nhidden,
    #               softplus_type=args.softplus_type,
    #               zero_softplus=args.zero_softplus,
    #               symm_act_first=args.symm_act_first) for _ in range(args.nblocks)]
    # layers = [None] * (2 * args.nblocks + 1)
    # layers[0::2] = [ActNorm(n_dims) for _ in range(args.nblocks + 1)]
    # layers[1::2] = [DeepConvexFlow(icnn, n_dims, unbiased=False,
    #                                atol=args.atol, rtol=args.rtol,
    #                                trainable_w0=args.trainable_w0) for _, icnn in zip(range(args.nblocks), icnns)]
    #
    # flow = SequentialFlow(layers)
    # flow = flow.to(device=device, dtype=dtype)
    # # deal with data-dependent initialization like actnorm.
    # with torch.no_grad():
    #     x = torch.rand(8, n_dims).to(device)
    #     flow.forward_transform(x)
    # map_location = lambda storage, loc: storage.cuda()
    # most_recent_path = torch.load(os.path.join(args.save, 'best_model.pth'), map_location=map_location)
    # flow.load_state_dict(most_recent_path['model'])
    # with open('wlatent1.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # flow.eval()
    # for f in flow.flows[1::2]:
    #     f.no_bruteforce = True
    # for param in flow.parameters():
    #     param.requires_grad = False
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
    with open('wlatent_face.pkl', 'rb') as f:
        data = pickle.load(f)
    flow.eval()
    for param in flow.parameters():
        param.requires_grad = False
    # ---------------------------------------------------------------------------------------------------

    g_ema = Generator(args.size, 512, 8).to(device)
    map_location = lambda storage, loc: storage.cuda()
    g_ema.load_state_dict(torch.load(args.ckpt, map_location=map_location)["g_ema"], strict=False)
    g_ema.eval()
    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda"), gpu_ids=[int(gpu_num)]
    )

    noises_single = g_ema.make_noise()
    noises = []  # stores all of the noise tensors
    noise_vars = []  # stores the noise tensors that we want to optimize on
    # num_trainable_noise_layers = g_ema.n_latent
    num_trainable_noise_layers = 0
    for i in range(g_ema.n_latent - 1):
        # dimension of the ith noise tensor
        res = (1, 1, 2 ** ((i + 1) // 2 + 2), 2 ** ((i + 1) // 2 + 2)) 
        new_noise = torch.randn(res, dtype=torch.float, device=device)
        if i < num_trainable_noise_layers:
            new_noise.requires_grad = True
            noise_vars.append(new_noise)
        else:
            new_noise.requires_grad = False
        noises.append(new_noise)

    with torch.no_grad():
        latent = torch.randn((n_mean_latent, 512), dtype=torch.float32, device=device)
        latent_out = g_ema.style(latent)
        latent_mean = latent_out.mean(0)
        gaussian_fit = {"mean": latent_out.mean(0).to(device), "std": latent_out.std(0).to(device)}

    # if args.w_plus:
    #     latent = g_ema.style(torch.randn((imgs.shape[0], 512), dtype=torch.float32, device=device))
    #     latent_in = latent.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)
    #     latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
    # else:
    #     latent = torch.randn((imgs.shape[0], 512), dtype=torch.float32, device=device)
    #     latent_in = g_ema.style(latent).detach().clone()
    if args.w_plus:
        latent = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)
        latent_in = latent.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
    else:
        latent_in = latent_mean.detach().clone().repeat(imgs.shape[0], 1)

    latent_in.requires_grad = True
    opt_dict = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax
    }
    # latent = latent_in
    optimizer = torch.optim.Adam([latent_in] + noise_vars, lr=args.lr)
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
    pbar = tqdm(range(args.steps))
    latent_path = []
    start_t = time.time()
    min_loss = np.inf

    x = latent_in.view(-1, latent_in.size(-1))
    # x = torch.nn.LeakyReLU(5)(x)
    x = (x - torch.from_numpy(data["mu"]).to(x)) / torch.from_numpy(data["std"]).to(x)
    logp = flow.log_probs(x, None).mean()
    print("logp loss latent_in: ", -logp.item())
    for i in pbar:
        t = i / args.steps
        optimizer.zero_grad()
        img_gen, _ = g_ema([latent_in], input_is_latent=True, noise=noises)
        # #-----NF ---------------------------------------------------------------------------
        p_loss = percept(img_gen, imgs).mean()
        mse_loss = F.l1_loss(img_gen, imgs)

        # latent_ = latent_in
        x = latent_in.view(-1, latent_in.size(-1))
        # x = torch.nn.LeakyReLU(5)(x)
        x = (x - torch.from_numpy(data["mu"]).to(x)) / torch.from_numpy(data["std"]).to(x)
        logp = flow.log_probs(x, None).mean()
        logp_loss = -logp
        a = (torch.ones(1) * math.sqrt(512)).to(device)
        p_norm_loss = torch.pow(torch.mean(torch.norm(latent_in, dim=-1)) - a, 2)
        loss = 5 * p_loss + 1 * mse_loss + 0.0001 * logp_loss
        cross_loss = 0
        if args.w_plus:
            cross_loss = cross(latent_in)
            loss += 0.01 * cross_loss
        if loss < min_loss:
            min_loss = loss
            best_summary = f'Percept: {p_loss.item():.4f};L1: {mse_loss.item():.4f}; Cross: {cross_loss:.4f};' \
                           f'logp_loss: {logp_loss.item()}; p_norm_loss: {p_norm_loss.item():.4f},'
            best_im = img_gen.detach().clone()
            best_latent = latent_in.detach().clone()
            best_noises = noises
        loss.backward()
        optimizer.step()
        scheduler.step()
        # noise_normalize_(noises)

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description(
            (
                f"Perc: {p_loss.item():.3f};"
                f" L1: {mse_loss.item():.3f};"
                f" logp_loss: {logp_loss:.3f};"
                f" cross: {cross_loss:.3f};"
                f" p_norm_loss: {p_norm_loss.item():.3f},"

            )
        )
        # print("\n", logp_loss, "\n")
        if i == args.steps - 1:
            x = best_latent.view(-1, latent_in.size(-1))
            # x = torch.nn.LeakyReLU(5)(x)
            x = (x - torch.from_numpy(data["mu"]).to(x)) / torch.from_numpy(data["std"]).to(x)
            logp = flow.log_probs(x, None).mean().item()
            print("logp loss latent_in: ", -logp)
    total_t = time.time() - start_t
    print(f'time: {total_t:.1f}')
    img_gen, _ = g_ema([best_latent], input_is_latent=True, noise=noises)
    filename = os.path.splitext(os.path.basename(args.files[0]))[0] + ".pt"
    img_ar = make_image(img_gen)
    torch.save(best_latent, "w_nf_plus_o")
    # torch.save(noises, "w_nf_n")
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

        img_name = os.path.splitext(os.path.basename(input_name))[0] + f'nf_{args.w_plus}_0001.png'
        pil_img = Image.fromarray(img_ar[i])
        pil_img.save(f'input/project/{img_name}')
        print(best_summary)
