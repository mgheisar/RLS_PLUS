# # implementation of PULSE with NF Gaussianization
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import argparse
import torchvision
import pickle
import torch
from torch.nn import functional as F
from tqdm import tqdm
from model import Generator
import time
import numpy as np
from FLOWS import flows as fnn
import math
from bicubic import BicubicDownSample
import glob
from torch.utils.data import DataLoader
from data_utils import Images

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def project_onto_l1_ball(x, eps):
    """
    See: https://gist.github.com/tonyduan/1329998205d88c566588e57e3e2c0c55
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


def cross(latent):
    if len(latent.shape) == 2:
        return 0
    DD = 0
    for i in range(latent.shape[0]):
        latent_ = latent[i, :].unsqueeze(0)
        # X = latent_.view(-1, 1, latent_.shape[1], latent_.shape[2])
        # Y = latent_.view(-1, latent_.shape[1], 1, latent_.shape[2])
        # A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
        A = torch.cdist(latent_, latent_, p=2)
        D = torch.sum(torch.triu(A, diagonal=1)) / ((latent_.shape[1] - 1) * latent_.shape[1] / 2)
        DD += D

    return DD / latent.shape[0]


toPIL = torchvision.transforms.ToPILImage()

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
    parser.add_argument("--clas", type=int, default=None, help="class label for the generator")
    parser.add_argument("--input_dir", type=str, default="input/project/lrr/lrrrr", help="path to the input image")
    parser.add_argument("--out_dir", type=str, default="input/project/resSR/RLSPlus", help="path to the output image")
    parser.add_argument('--factor', type=int, default=16, help='Super resolution factor')
    parser.add_argument("--gpu_num", type=int, default=1, help="gpu number")
    parser.add_argument("--duplicate", type=int, default=1, help='number of duplications')
    parser.add_argument('--augs', default=None, nargs='+', help='which augmentations are used to test robustness',
                        choices=['rotate', 'vflip', 'hflip', 'contrast', 'brightness', 'gaussiannoise',
                                 'occlusion',
                                 'regularblur', 'defocusblur', 'motionblur', 'gaussianblur', 'saltpepper',
                                 'perspective', 'gray', 'colorjitter'])
    parser.add_argument("--save_anchor", action="store_false", help="allow to save anchor points")
    # ---------------------------------------------------
    parser.add_argument("--steps", type=int, default=500, help="optimize iterations")
    parser.add_argument("--lr", type=float, default=0.5, help="learning rate")
    parser.add_argument('--logp', type=float, default=0.002, help='logp regularization')  # 0.001
    parser.add_argument('--pnorm', type=float, default=0.004, help='pnorm regularization')  # 0.002
    parser.add_argument('--cross', type=float, default=0.5, help='cross regularization')
    parser.add_argument("--w_plus", action="store_false", help="allow to use distinct latent codes to each layers")
    parser.add_argument("--batchsize", type=int, default=1, help="batch size")
    parser.add_argument('--eps', type=float, default=0.5)
    # ---------------------------------------------------
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
    # parser.add_argument(
    #     "--files", metavar="FILES", nargs="+", help="path to image files to be projected"
    # )
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
                        default='checkpoint/maf_face1024')
    parser.add_argument("--nf_stat", type=str, default='wlatent_face1024.pkl', help='latent data stats')
    parser.add_argument('--dimh', type=int, default=4096)  # 64:img
    parser.add_argument('--nhidden', type=int, default=5)  # 4:img
    parser.add_argument("--nblocks", type=int, default=10, help='Number of stacked CPFs.')  # 8-8-8 img
    parser.add_argument('--atol', type=float, default=1e-3)
    parser.add_argument('--rtol', type=float, default=0.0)
    parser.add_argument('--fp64', action='store_true', default=False)
    parser.add_argument('--brute_val', action='store_true', default=False)

    args = parser.parse_args()
    if args.augs is not None:
        args.out_dir = os.path.join(args.out_dir, args.augs[0])
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    # gpu_num = args.gpu_num
    # torch.cuda.set_device(gpu_num)
    # cuda = torch.cuda.is_available()

    n_mean_latent = 1000000
    if args.clas is None:
        image_list = sorted(glob.glob(f"{args.input_dir}/*_{args.factor}x.jpg"))[:1000]
    else:
        image_list = sorted(glob.glob(f"{args.input_dir}/{args.clas}/*.png"))
    dataset = Images(image_list, duplicates=args.duplicate, aug=args.augs, factor=args.factor)
    dataloader = DataLoader(dataset, batch_size=args.batchsize)
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
    best_model_path = torch.load(os.path.join(args.save, 'best_model.pth'), map_location=device)
    flow.load_state_dict(best_model_path['model'])
    flow.to(device)
    with open(args.nf_stat, 'rb') as f:
        data = pickle.load(f)
    flow.eval()
    for param in flow.parameters():
        param.requires_grad = False
    # ---------------------------------------------------------------------------------------------------

    g_ema = Generator(args.size, 512, 8).to(device)
    g_ema.load_state_dict(torch.load(args.ckpt, map_location=device)["g_ema"], strict=False)
    g_ema.eval()

    # discriminator = Discriminator(args.size).to(device)
    #
    # discriminator.load_state_dict(torch.load(args.ckpt, map_location=device)["d"], strict=False)
    # discriminator.eval()
    # percept = lpips.PerceptualLoss(
    #     model="net-lin", net="vgg", use_gpu=device.startswith("cuda"), gpu_ids=[int(gpu_num)]
    # )
    Downsampler = BicubicDownSample(factor=args.factor)

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

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    image_index = 0
    for ref_im, ref_im_hr, ref_im_name in dataloader:
        image_id = ref_im_name[0].split("_")[0]
        if args.duplicate == 1:
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True
            if args.w_plus:
                latent = latent_mean.detach().clone().unsqueeze(0).repeat(args.batchsize, 1)
                latent_in = latent.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
            else:
                latent_in = latent_mean.detach().clone().repeat(args.batchsize, 1)
        else:
            if args.w_plus:
                latent = g_ema.style(torch.randn((args.batchsize, 512), dtype=torch.float32, device=device))
                latent = latent_mean + 0.2 * (latent - latent_mean)  # 0.4
                latent_in = latent.unsqueeze(1).repeat(1, g_ema.n_latent, 1).detach().clone()
            else:
                latent = g_ema.style(torch.randn((args.batchsize, 512), dtype=torch.float32, device=device))
                latent = latent_mean + 0.2 * (latent - latent_mean)  # 0.4
                latent_in = latent.detach().clone()
            if image_index % args.duplicate == 0:
                best_latent_multiple = []

        # latent_mean_ = torch.load('w_nf_plus_o', map_location=device)
        # latent_in = latent_mean_.detach().clone()

        latent_in.requires_grad = True
        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adamax': torch.optim.Adamax
        }
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
        start_t = time.time()
        min_loss = np.inf
        for i in pbar:
            t = i / args.steps
            optimizer.zero_grad()
            img_gen, _ = g_ema([latent_in], input_is_latent=True, noise=noises)
            img_gen = (img_gen + 1) / 2
            # #-----NF ---------------------------------------------------------------------------
            l1_loss = F.l1_loss(Downsampler(img_gen), ref_im)
            mse_loss = F.mse_loss(Downsampler(img_gen), ref_im)
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
            p_norm_loss = torch.mean(torch.pow(torch.norm(x, dim=-1) - a, 2))

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
                best_latent = latent_in.detach().clone()
                best_step = i + 1
                best_rec = l1_loss.item()
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

            pbar.set_description(
                (
                    f" L1: {l1_loss.item():.3f};"
                    f" logp: {logp_loss:.3f};"
                    f" cross: {cross_loss:.3f};"
                    f" pn: {p_norm_loss.item():.3f},"

                )
            )
        if best_rec > args.eps:
            print("Generated image might not be satisfactory. Try running the search loop again.")
        else:
            if args.augs is not None and args.save_anchor:
                torch.save(best_latent, f'{args.out_dir}/wnf_{image_id}')
            elif args.save_anchor:
                if args.duplicate == 1:
                    torch.save(best_latent, f'input/project/resSR/RLSPlus/wnf_{args.factor}/wnf_{image_id}+')
                else:
                    best_latent_multiple.append(best_latent)
                    image_index += 1
                    if image_index % args.duplicate == 0:
                        torch.save(best_latent_multiple, f'input/project/resSR/RLSPlus/wnf_{args.factor}/'
                                                         f'w_nf_{image_id}_{args.duplicate}+')

            total_t = time.time() - start_t
            print(f'time: {total_t:.1f}')
            best_im_LR = Downsampler(best_im)
            # perceptual = percept(best_im, ref_im_hr).mean()
            # L1_norm = F.l1_loss(best_im, ref_im_hr).mean()
            for i in range(args.batchsize):
                pil_img = toPIL(best_im[i].cpu().detach().clamp(0, 1))
                pil_img_lr = toPIL(best_im_LR[i].cpu().detach().clamp(0, 1))
                # img_name = ref_im_name[i] + f'boost.jpg'
                # img_name = ref_im_name[i] + 'lr' + str(args.lr).split('.')[-1] \
                #            + '_logp' + str(args.logp).split('.')[-1] + '_cross' + str(args.cross).split('.')[-1] \
                #            + '_pnorm' + str(args.pnorm).split('.')[-1] + '_step' + str(best_step) + '.jpg'
                if args.clas is None:
                    img_name = f'{ref_im_name[i]}.jpg'
                    pil_img.save(f'{args.out_dir}/{img_name}')
                else:
                    img_name = f'{ref_im_name[i]}.png'
                    pil_img.save(f'{args.out_dir}/{args.clas}/{img_name}')
                # pil_img = toPIL(ref_im_hr[i].cpu().detach().clamp(0, 1))
                # img_name = f'{ref_im_name[i]}_HR.jpg'
                # pil_img.save(f'input/project/{img_name}')

            print(best_summary)
            # print(' percept: ', perceptual.item(), 'l1:', L1_norm.item())
