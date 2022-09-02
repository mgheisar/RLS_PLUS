# # implementation of PULSE with NF Gaussianization
import os
import argparse
import torchvision
import dnnlib
import torch
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import lpips
from model import Generator
import time
import numpy as np
from bicubic import BicubicDownSample
import glob
from torch.utils.data import DataLoader, Dataset
from torch_utils_brgm.forwardModels import *


class Images(Dataset):
    def __init__(self, image_list, duplicates):
        # args.files = [sorted(glob.glob(f"input/project/inputt/*_{args.factor}x.jpg"))[args.img_idx]]
        self.image_list = image_list
        self.duplicates = duplicates  # Number of times to duplicate the image in the dataset to produce multiple HR
        # self.transform = torchvision.transforms.Compose([transforms.ToTensor()])
        # , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    @staticmethod
    def transform(image):
        image = torch.tensor(np.array(image).transpose([2, 0, 1]), dtype=torch.float32)
        return image

    def __len__(self):
        return self.duplicates * len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx // self.duplicates]
        image = self.transform(Image.open(img_path)).to(torch.device("cuda"))
        hr_path = "input/project/resHR/" + os.path.basename(img_path).split(".")[0] + "_HR.jpg"
        image_hr = self.transform(Image.open(hr_path)).to(torch.device("cuda"))
        if self.duplicates == 1:
            return image, image_hr, os.path.splitext(os.path.basename(img_path))[0]
        else:
            return image, image_hr, os.path.splitext(os.path.basename(img_path))[0] + f"_{(idx % self.duplicates) + 1}"


def getVggFeatures(images, num_channels, vgg16):
    # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
    # if synth_images_down.shape[2] > 256:
    images_vgg = F.interpolate(images, size=(256, 256), mode='area')

    if num_channels == 1:
        # if grayscale, move back to RGB to evaluate perceptual loss
        images_vgg = images_vgg.repeat(1, 3, 1, 1)  # BCWH

    # Features for synth images.
    features = vgg16(images_vgg, resize_images=False, return_lpips=True)
    return features


def cosine_distance(latentsBLD):
    # assert latentsBLD.shape[0] == 1
    cosDist = 0
    for b in range(latentsBLD.shape[0]):
        latentsNormLD = F.normalize(latentsBLD[0, :, :], dim=1, p=2)
        cosDistLL = 1 - torch.matmul(latentsNormLD, latentsNormLD.T)
        cosDist += cosDistLL.reshape(-1).norm(p=1)
    return cosDist


def toPIL(image):
    image = image.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy().squeeze()
    return Image.fromarray(image, 'RGB')

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
    # parser.add_argument(
    #     "--files", metavar="FILES", nargs="+", help="path to image files to be projected"
    # )
    parser.add_argument('--save_intermediate', action="store_true",
                        help='Whether to store and save intermediate images during optimization')
    parser.add_argument('--lr_schedule', type=str, default='linear1cycledrop',
                        help='fixed, linear1cycledrop, linear1cycle')

    # ---------------------------------------------------
    parser.add_argument("--input_dir", type=str, default="input/input", help="output directory")
    parser.add_argument('--factor', type=int, default=64, help='Super resolution factor')
    parser.add_argument("--steps", type=int, default=501, help="optimize iterations")
    parser.add_argument("--lr", type=float, default=0.5, help="learning rate")
    parser.add_argument('--logp', type=float, default=0.0005, help='logp regularization')
    parser.add_argument('--cross', type=float, default=0.1, help='cross regularization')
    parser.add_argument('--pnorm', type=float, default=0.01, help='pnorm regularization')
    parser.add_argument("--out_dir", type=str, default="", help="output directory")
    parser.add_argument("--gpu_num", type=int, default=1, help="gpu number")
    parser.add_argument("--batchsize", type=int, default=1, help="batch size")
    parser.add_argument('--eps', type=float, default=100)

    args = parser.parse_args()
    gpu_num = args.gpu_num
    torch.cuda.set_device(gpu_num)
    cuda = torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_mean_latent = 1000000
    image_list = sorted(glob.glob(f"input/project/resLR/*_{args.factor}x.jpg"))[:7327]  # 7327:14654
    dataset = Images(image_list, duplicates=1)
    dataloader = DataLoader(dataset, batch_size=args.batchsize)
    # ---------------------------------------------------------------------------------------------------
    lambda_pix = 0.001
    lambda_perc = 10000000
    lambda_w = 100
    lambda_c = 0.1  # tried many values, this is a good one for in-painting
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
    # Downsampler = BicubicDownSample(factor=args.factor, device=device)
    Downsampler = ForwardDownsample(factor=1/args.factor)

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
        w_std_scalar = torch.tensor((torch.sum((latent_out - latent_mean) ** 2) / n_mean_latent) ** 0.5)

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)
    for ref_im, ref_im_hr, ref_im_name in dataloader:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        target_features = getVggFeatures(ref_im, 3, vgg16)
        # if args.w_plus:
        #     latent = g_ema.style(torch.randn((args.batchsize, 512), dtype=torch.float32, device=device))
        #     latent = latent_mean + 0.25 * (latent - latent_mean)
        #     latent_in = latent.unsqueeze(1).repeat(1, g_ema.n_latent, 1).detach().clone()
        # else:
        #     latent = g_ema.style(torch.randn((args.batchsize, 512), dtype=torch.float32, device=device))
        #     latent = latent_mean + 0.25 * (latent - latent_mean)
        #     latent_in = latent.detach().clone()

        if args.w_plus:
            latent = latent_mean.detach().clone().unsqueeze(0).repeat(args.batchsize, 1)
            latent_in = latent.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
        else:
            latent_in = latent_mean.detach().clone().repeat(args.batchsize, 1)

        # latent_mean_ = torch.load('w_nf_plus_o', map_location=map_location)
        # latent_in = latent_mean_.detach().clone()

        latent_in.requires_grad = True
        optimizer = torch.optim.Adam([latent_in] + noise_vars, lr=args.lr)
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

        # optimizer = torch.optim.Adam([latent_in],
        #                              betas=(0.9, 0.999),
        #                              lr=0.1)
        pbar = tqdm(range(args.steps))
        start_t = time.time()
        min_loss = np.inf
        for i in pbar:
            t = i / args.steps
            optimizer.zero_grad()
            img_gen, _ = g_ema([latent_in], input_is_latent=True, noise=noises)
            img_gen = (img_gen + 1) * (255 / 2)
            # #-----NF ---------------------------------------------------------------------------
            img_gen_down = Downsampler(img_gen)
            pixelwise_loss = lambda_pix * (img_gen_down - ref_im).square().mean()
            loss = 0
            loss += pixelwise_loss

            # perceptual loss
            synth_features = getVggFeatures(img_gen_down, 3, vgg16)
            perceptual_loss = lambda_perc * (target_features - synth_features).square().mean()
            loss += perceptual_loss

            # adding prior on w ~ N(mu, sigma) as extra loss term
            w_loss = lambda_w * (
                    latent_in / w_std_scalar - latent_mean / w_std_scalar).square().mean()  # will broadcast w_avg: [1, 1, 512] to ws: [1, L, 512]
            loss += w_loss

            # adding cosine distance loss
            cosine_loss = lambda_c * cosine_distance(latent_in)
            loss += cosine_loss
            if loss < min_loss:
                min_loss = loss
                best_summary = f'L1: {pixelwise_loss.item():.3f}; L2: {perceptual_loss.item():.3f}; ' \
                               f'cross: {cosine_loss:.3f};w_loss: {w_loss:.3f}'
                best_im = img_gen.detach().clone()
                best_step = i + 1
                best_rec = pixelwise_loss.item()
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
                    f" L2: {pixelwise_loss.item():.3f};"
                    f" percept: {perceptual_loss:.3f};"
                    f" cross: {cosine_loss:.3f};"
                    f" w_loss: {w_loss:.3f};"
                )
            )
        if best_rec > args.eps:
            print("Generated image might not be satisfactory. Try running the search loop again.")
        else:
            total_t = time.time() - start_t
            print(f'time: {total_t:.1f}')
            best_im_LR = Downsampler(best_im)
            perceptual = percept(best_im, ref_im_hr).mean()
            L1_norm = F.l1_loss(best_im, ref_im_hr).mean()
            for i in range(args.batchsize):
                pil_img = toPIL(best_im[i])
                pil_img_lr = toPIL(best_im_LR[i])
                # img_name = ref_im_name[i] + f'boost.jpg'
                # img_name = ref_im_name[i] + 'lr' + str(args.lr).split('.')[-1] \
                #            + '_logp' + str(args.logp).split('.')[-1] + '_cross' + str(args.cross).split('.')[-1] \
                #            + '_pnorm' + str(args.pnorm).split('.')[-1] + '_step' + str(best_step) + '.jpg'
                img_name = f'{ref_im_name[i]}_boost_l1_{best_rec:.3f}.jpg'
                pil_img.save(f'input/project/{img_name}')
                pil_img = toPIL(ref_im_hr[i])
                img_name = f'{ref_im_name[i]}_HR.jpg'
                pil_img.save(f'input/project/{img_name}')

            print(best_summary)
            print(' percept: ', perceptual.item(), 'l1:', L1_norm.item())
