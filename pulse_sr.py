# # implementation of PULSE with NF Gaussianization
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torchvision
import pickle
from SphericalOptimizer import SphericalOptimizer
import torch
from torch import optim
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm
import lpips
from model import Generator
import time
import numpy as np
from FLOWS import flows as fnn
import glob
import gc
from bicubic import BicubicDownSample
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


def geocross(latent):
    if (latent.shape[1] == 1):
        return 0
    else:
        X = latent.view(-1, 1, latent.shape[1], latent.shape[2])
        Y = latent.view(-1, latent.shape[1], 1, latent.shape[2])
        A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
        B = ((X + Y).pow(2).sum(-1) + 1e-9).sqrt()
        D = 2 * torch.atan2(A, B)
        D = ((D.pow(2) * latent.shape[2]).mean((1, 2)) / 8.).sum()
        # D = ((D.pow(2) * latent.shape[2]).mean((1, 2)) / latent.shape[1]).sum()
        return D


class Images(Dataset):
    def __init__(self, image_list, duplicates):
        # args.files = [sorted(glob.glob(f"input/project/inputt/*_{args.factor}x.jpg"))[args.img_idx]]
        self.image_list = image_list
        self.duplicates = duplicates  # Number of times to duplicate the image in the dataset to produce multiple HR images

    def __len__(self):
        return self.duplicates * len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx // self.duplicates]
        image = torchvision.transforms.ToTensor()(Image.open(img_path)).to(torch.device("cuda"))
        image_hr = []
        # hr_path = "input/project/resHR/" + os.path.basename(img_path).split(".")[0] + "_HR.jpg"
        # image_hr = torchvision.transforms.ToTensor()(Image.open(hr_path)).to(torch.device("cuda"))
        # image_hr = torchvision.transforms.ToTensor()(Image.open(img_path.split('_')[0] + '_HR.jpg')).to(
        # torch.device("cuda"))
        if self.duplicates == 1:
            return image, image_hr, os.path.splitext(os.path.basename(img_path))[0]
        else:
            return image, image_hr, os.path.splitext(os.path.basename(img_path))[0] + f"_{(idx % self.duplicates) + 1}"


toPIL = torchvision.transforms.ToPILImage()

if __name__ == "__main__": # run sr_boost script
    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--size", type=int, default=1024, help="output image sizes of the generator"
    )
    parser.add_argument("--clas", type=int, default=0, help="class label for the generator")
    parser.add_argument("--input_dir", type=str, default="input/project", help="path to the input image")
    parser.add_argument("--out_dir", type=str, default="input/project", help="path to the output image")
    parser.add_argument('--factor', type=int, default=64, help='Super resolution factor')
    parser.add_argument("--gpu_num", type=int, default=0, help="gpu number")
    parser.add_argument("--duplicate", type=int, default=1, help="number of times to duplicate the image in the dataset")
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
    parser.add_argument("--lr", type=float, default=0.4, help="learning rate")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--steps", type=int, default=500, help="optimize iterations")
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
    # parser.add_argument(
    #     "files", metavar="FILES", nargs="+", help="path to image files to be projected"
    # )
    parser.add_argument('--save_intermediate', action="store_true",
                        help='Whether to store and save intermediate images during optimization')
    parser.add_argument('--lr_schedule', type=str, default='linear1cycledrop',
                        help='fixed, linear1cycledrop, linear1cycle')
    parser.add_argument('--img_idx', type=int, default=2, help='image index')
    parser.add_argument('--eps', type=float, default=10, help='epsilon')
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
    parser.add_argument('--dimh', type=int, default=4096)  # 64:img
    parser.add_argument('--nhidden', type=int, default=5)  # 4:img
    parser.add_argument("--nblocks", type=int, default=10, help='Number of stacked CPFs.')  # 8-8-8 img
    parser.add_argument('--atol', type=float, default=1e-3)
    parser.add_argument('--rtol', type=float, default=0.0)
    parser.add_argument('--fp64', action='store_true', default=False)
    parser.add_argument('--brute_val', action='store_true', default=False)
    parser.add_argument("--batchsize", type=int, default=1, help="batch size")

    args = parser.parse_args()
    gpu_num = args.gpu_num
    torch.cuda.set_device(gpu_num)
    cuda = torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_mean_latent = 1000000
    # image_list = sorted(glob.glob(f"input/project/lrr/*_{args.factor}x.jpg"))
    image_list = sorted(glob.glob(f"{args.input_dir}/{args.clas}/*.png"))
    dataset = Images(image_list, duplicates=args.duplicate)
    dataloader = DataLoader(dataset, batch_size=args.batchsize)
    # # # # -------Loading NF model----------------------------------------------------------------------------
    # num_blocks = 5
    # num_inputs = 512
    # modules = []
    # for _ in range(num_blocks):
    #     modules += [
    #         fnn.MADE(num_inputs, num_hidden=1024, num_cond_inputs=None, act='relu'),
    #         fnn.BatchNormFlow(num_inputs),
    #         fnn.Reverse(num_inputs)
    #     ]
    # flow = fnn.FlowSequential(*modules)
    # map_location = lambda storage, loc: storage.cuda()
    # best_model_path = torch.load(os.path.join(args.save, 'best_model.pth'), map_location=map_location)
    # flow.load_state_dict(best_model_path['model'])
    # flow.to(device)
    # with open(args.nf_stat, 'rb') as f:
    #     data = pickle.load(f)
    # flow.eval()
    # for param in flow.parameters():
    #     param.requires_grad = False
    # ---------------------------------------------------------------------------------------------------

    g_ema = Generator(args.size, 512, 8).to(device)
    map_location = lambda storage, loc: storage.cuda()
    g_ema.load_state_dict(torch.load(args.ckpt, map_location=map_location)["g_ema"], strict=False)
    g_ema.eval()
    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda"), gpu_ids=[int(gpu_num)]
    )
    Downsampler = BicubicDownSample(factor=args.factor, device=device)
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
        torch.manual_seed(0)
        latent = torch.randn((n_mean_latent, 512), dtype=torch.float32, device=device)
        latent_out = torch.nn.LeakyReLU(5)(g_ema.style(latent))
        latent_mean = torch.nn.LeakyReLU(5)(g_ema.style(latent).mean(0))
        gaussian_fit = {"mean": latent_out.mean(0).to(device), "std": latent_out.std(0).to(device)}

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    for ref_im, ref_im_hr, ref_im_name in dataloader:
        if args.duplicate == 1:
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True
        if args.w_plus:
            latent = torch.randn(
                (args.batchsize, g_ema.n_latent, 512), dtype=torch.float32, device=device)
        else:
            latent = torch.randn((args.batchsize, 512), dtype=torch.float32, device=device)

        # if args.w_plus:
        #     latent_ = latent_mean.detach().clone().unsqueeze(0).repeat(args.batchsize, 1)
        #     latent = latent_.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
        # else:
        #     latent = latent_mean.detach().clone().repeat(args.batchsize, 1)

        latent.requires_grad = True
        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'adamax': torch.optim.Adamax,
        }
        # latent = latent_in
        # optimizer = torch.optim.Adam([latent_in] + noise_vars, lr=args.lr)
        optimizer = SphericalOptimizer(optim.Adam, [latent] + noise_vars, lr=args.lr)
        schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9 * (1 - np.abs(x / args.steps - 1 / 2) * 2) + 1) / 10,
            'linear1cycledrop': lambda x: (9 * (
                    1 - np.abs(
                x / (0.9 * args.steps) - 1 / 2) * 2) + 1) / 10 if x < 0.9 * args.steps else 1 / 10 + (
                    x - 0.9 * args.steps) / (0.1 * args.steps) * (1 / 1000 - 1 / 10),
        }
        schedule_func = schedule_dict[args.lr_schedule]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer.opt, schedule_func)
        pbar = tqdm(range(args.steps))
        start_t = time.time()
        min_loss = np.inf
        for i in pbar:
            t = i / args.steps
            optimizer.opt.zero_grad()
            latent_in = latent
            latent_in = torch.nn.LeakyReLU(negative_slope=0.2)(latent_in * gaussian_fit["std"] + gaussian_fit["mean"])
            img_gen, _ = g_ema([latent_in], input_is_latent=True, noise=noises)
            img_gen = (img_gen + 1) / 2
            # #-----NF ---------------------------------------------------------------------------
            # p_loss = percept(img_gen, ref_im).mean()
            mse_loss = F.mse_loss(Downsampler(img_gen), ref_im)
            l1_loss = F.l1_loss(Downsampler(img_gen), ref_im)
            # # latent_ = latent_in
            # x = latent_in.view(-1, latent_in.size(-1))
            # x = (x - torch.from_numpy(data["mu"]).to(x)) / torch.from_numpy(data["std"]).to(x)
            # logp = flow.log_probs(x, None).mean()
            # logp_loss = -logp
            # a = (torch.ones(1) * math.sqrt(512)).to(device)
            # p_norm_loss = torch.pow(torch.mean(torch.norm(latent_in, dim=-1)) - a, 2)
            loss = 100 * mse_loss
            cross_loss = 0
            if args.w_plus:
                cross_loss = geocross(latent)
                loss += 0.1 * cross_loss  # 0.01
            if loss < min_loss:
                min_loss = loss
                best_summary = f'L1: {l1_loss.item():.3f}; L2: {mse_loss.item():.3f}; Cross: {cross_loss:.3f};'
                best_im = img_gen.detach().clone()
                best_rec = l1_loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.set_description(
                (
                    # f"Perc: {p_loss.item():.3f};"
                    f" L1: {l1_loss.item():.3f};"
                    # f" logp_loss: {logp_loss:.3f};"
                    f" cross: {cross_loss:.3f};"
                    # f" p_norm_loss: {p_norm_loss.item():.3f},"

                )
            )
            # if i == args.steps - 1:
            #     x = latent_in.view(-1, latent_in.size(-1))
            #     x = (x - torch.from_numpy(data["mu"]).to(x)) / torch.from_numpy(data["std"]).to(x)
            #     logp = flow.log_probs(x, None).mean().item()
            #     print("logp loss latent_in: ", -logp)
        if best_rec > args.eps:
            print("Generated image might not be satisfactory. Try running the search loop again.")
        else:
            total_t = time.time() - start_t
            print(f'time: {total_t:.1f}')
            best_im_LR = Downsampler(best_im)
            # perceptual = percept(best_im, ref_im_hr).mean()
            # L1_norm = F.l1_loss(best_im, ref_im_hr).mean()
            for i in range(args.batchsize):
                pil_img = toPIL(best_im[i].cpu().detach().clamp(0, 1))
                pil_img_lr = toPIL(best_im_LR[i].cpu().detach().clamp(0, 1))
                # torch.save(best_latent, "w_nfp")
                # torch.save(noises, "w_nfp_n")
                # img_name = f'{ref_im_name[i]}_pulse_l1_{best_rec:.3f}.jpg'
                img_name = f'{ref_im_name[i]}_1.png'
                pil_img.save(f'{args.out_dir}/{args.clas}/{img_name}')
                # pil_img = toPIL(ref_im_hr[i].cpu().detach().clamp(0, 1))
                # img_name = f'{ref_im_name[i]}_HR.jpg'
                # pil_img.save(f'{args.out_dir}/{img_name}')
            print(best_summary)
            # print(' percept: ', perceptual.item(), 'l1:', L1_norm.item())