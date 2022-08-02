# Running I2S+Adv for Super resolution
from model import Generator, Discriminator
import numpy as np
import time
from loss_projection import LossBuilder
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch
from PIL import Image
import argparse
from torchvision import transforms
import math
import os
import glob
from math import log10, ceil
from tqdm import tqdm
import gc

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()
        noise.data.add_(-mean).div_(std)


def normalize_image_(img):  # image(b,C,H,W)
    if len(img.size()) == 4:
        image = img.view(img.size(0), img.size(1), -1)
    else:
        image = img.view(img.size(0), -1)
    image -= image.min(2, keepdim=True)[0]
    image /= image.max(2, keepdim=True)[0]
    image -= 0.5
    image /= 0.5
    image = image.view(*img.size())
    return image


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength
    return latent + noise


def make_image(tensor):
    return (tensor.detach().clamp_(min=-1, max=1).add(1).div_(2).mul(255).type(torch.uint8).
            permute(1, 2, 0).to("cpu").numpy())


class Images(Dataset):
    def __init__(self, root_dir, duplicates, transform):
        self.root_path = Path(root_dir)
        self.image_list = list(self.root_path.glob("*.png"))
        self.duplicates = duplicates  # Number of times to duplicate the image in the dataset to produce multiple HR images
        self.transform = transform

    def __len__(self):
        return self.duplicates * len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx // self.duplicates]
        image = self.transform(Image.open(img_path).convert("RGB")).to(device)
        if self.duplicates == 1:
            return image, img_path.stem
        else:
            return image, img_path.stem + f"_{(idx % self.duplicates) + 1}"


class PULSE(torch.nn.Module):
    def __init__(self, args, device, normalize=False, verbose=True):
        super(PULSE, self).__init__()
        self.verbose = verbose
        self.device = device
        if self.verbose: print("Loading Synthesis Network")
        self.g_ema = Generator(args.size, 512, 8).to(self.device)
        self.discriminator = Discriminator(args.size).to(self.device)
        self.map_location = lambda storage, loc: storage.cuda()
        gan = torch.load(args.ckpt, map_location=self.map_location)
        self.g_ema.load_state_dict(gan["g_ema"], strict=False)
        self.g_ema.eval()
        self.discriminator.load_state_dict(gan["d"], strict=False)
        self.discriminator.eval()
        self.args = args
        self.normalize = normalize
        del gan
        torch.cuda.empty_cache()
        n_mean_latent = 1000000
        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, 512, device=device)
            latent_out = self.g_ema.style(noise_sample)

            self.latent_mean = latent_out.mean(0)
            self.latent_std = ((latent_out - self.latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    def forward(self, ref_im):
        # torch.manual_seed(1)
        # torch.cuda.manual_seed(1)
        # torch.backends.cudnn.deterministic = True
        batch_size = ref_im.shape[0]
        # ref_pred = self.discriminator(ref_im)
        ref_pred = None
        # # ------------------------------------------------------------AA
        noises_single = self.g_ema.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(batch_size, 1, 1, 1).normal_())
        for i, noise in enumerate(noises):
            if i < self.args.num_trainable_noise_layers:
                noise.requires_grad = True
            else:
                noise.requires_grad = False

        # if self.args.w_plus:
        #     latent_in = self.latent_mean.detach().clone().unsqueeze(0).repeat(batch_size, 1)
        #     latent_in = latent_in.unsqueeze(1).repeat(1, self.g_ema.n_latent, 1)
        # else:
        #     latent_in = self.latent_mean.detach().clone().unsqueeze(0).repeat(batch_size, 1, 1)
        if args.w_plus:
            latent_in = self.latent_mean.detach().clone().unsqueeze(0).repeat(batch_size, 1)
            latent_in = latent_in.unsqueeze(1).repeat(1, self.g_ema.n_latent, 1)
        else:
            latent_in = self.latent_mean.detach().clone().repeat(batch_size, 1)
        latent_in.requires_grad = True
        var_list = [latent_in] + noises
        # # ------------------------------------------------------------AA

        # # ------------------------------------------------------------BB
        opt = torch.optim.Adam(var_list, lr=self.args.learning_rate)
        schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9 * (1 - np.abs(x / self.args.steps - 1 / 2) * 2) + 1) / 10,
            'linear1cycledrop': lambda x: (9 * (
                    1 - np.abs(
                x / (0.9 * self.args.steps) - 1 / 2) * 2) + 1) / 10 if x < 0.9 * self.args.steps else 1 / 10 + (
                    x - 0.9 * self.args.steps) / (0.1 * self.args.steps) * (1 / 1000 - 1 / 10),
        }
        schedule_func = schedule_dict[self.args.lr_schedule]
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule_func)
        # # -------------------------------------------------------------BB
        loss_builder = LossBuilder(ref_im, ref_pred, self.args.loss_str,
                                   self.args.factor, device=self.device,
                                   normalize=self.normalize, gpu_num=self.args.gpu_num)

        min_loss = np.inf
        best_summary = ""
        start_t = time.time()
        gen_im = None

        if self.verbose:
            print("Optimizing")
        pbar = tqdm(range(self.args.steps))
        for j in pbar:
            t = j / self.args.steps
            # # ------------------------------------------------------------BB
            # lr = get_lr(t, self.args.learning_rate, self.args.lr_rampdown)
            # opt.param_groups[0]["lr"] = lr
            opt.zero_grad()
            # # ------------------------------------------------------------BB
            # if self.args.w_plus:
            #     latent_n = latent_in
            # else:
            #     latent_n = latent_in.expand(-1, self.g_ema.n_latent, -1)
            # # ------------------------------------------------------------AA
            # latent_n = latent_in
            # noise_strength = self.latent_std * self.args.noise * max(0, 1 - t / self.args.noise_ramp) ** 2
            # latent_n = latent_noise(latent_n, noise_strength.item())
            gen_im, _ = self.g_ema([latent_in], input_is_latent=True, noise=noises)
            # # ------------------------------------------------------------AA
            # Calculate Losses
            fake_pred = self.discriminator(gen_im)
            loss, loss_dict = loss_builder(latent_in, gen_im, fake_pred, noises)
            loss_dict['TOTAL'] = loss
            if j == 0 and self.verbose:
                summary = f'First iteration ({j + 1}) | ' + ' | '.join(
                    [f'{x}: {y:.4f}' for x, y in loss_dict.items()])
                print(summary)
            # Save best summary for log
            if loss < min_loss:
                min_loss = loss
                best_summary = f'BEST ({j + 1}) | ' + ' | '.join(
                    [f'{x}: {y:.4f}' for x, y in loss_dict.items()])
                best_im = gen_im.detach().clone()
                # best_latent = latent_in.detach().clone()
                # best_noises = noises

            if self.args.save_intermediate:
                yield best_im.clone().cpu().detach(), loss_builder.D(best_im).clone().cpu().detach()

            loss.backward()
            opt.step()
            scheduler.step()
            noise_normalize_(noises)
            pbar.set_description(
                (f'First iteration ({j + 1}) | ' + ' | '.join(
                    [f'{x}: {y:.4f}' for x, y in loss_dict.items()])
                )
            )

        total_t = time.time() - start_t
        current_info = f' | time: {total_t:.1f}'
        if self.verbose:
            print(best_summary + current_info)
        yield best_im, loss_builder.D(best_im).detach().clone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projection with PULSE")
    # I/O arguments
    parser.add_argument('--input_dir', type=str, default='input/superres/LR/0', help='input data directory')
    parser.add_argument('--output_dir', type=str, default='input/superres/SR/0', help='output data directory')
    parser.add_argument('--duplicates', type=int, default=1,
                        help='How many HR images to produce for every image in the input directory')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size to use during optimization')
    parser.add_argument("--size", type=int, default=128, help="output image size")
    # PULSE arguments
    parser.add_argument("--ckpt", type=str, default="checkpoint/Golgi/250000.pt", help="path to the model checkpoint")
    parser.add_argument("--w_plus", action="store_false", help="allow to use distinct latent codes to each layers")
    parser.add_argument('--save_intermediate', action="store_true",
                        help='Whether to store and save intermediate images during optimization')
    parser.add_argument('-num_trainable_noise_layers', type=int, default=12, help='Number of noise layers to optimize')
    parser.add_argument("--noise", type=float, default=0.05, help="strength of the noise level")
    parser.add_argument("--noise_ramp", type=float, default=0.75, help="duration of the noise level decay", )

    parser.add_argument('--factor', type=int, default=8, help='Super resolution factor')
    parser.add_argument('--steps', type=int, default=1000, help='Number of optimization steps')
    parser.add_argument("--lr_rampdown", type=float, default=0.5, help="duration of the learning rate decay")
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate to use during optimization')
    parser.add_argument("--gpu_num", type=int, default=2, help="gpu number")
    parser.add_argument('--loss_str', type=str, default="1*L2", help='Loss function to use')  # Golgi "10*L1+1*Adv"
    parser.add_argument('--lr_schedule', type=str, default='linear1cycledrop',
                        help='fixed, linear1cycledrop, linear1cycle')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_num)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    # args.input_dir = 'input/superres/LR_8x/1'
    # args.output_dir = 'input/superres/SR_8x_I2S/1'
    files = sorted(glob.glob(os.path.join(args.input_dir, '*.png')))
    output_dir = ""
    dataset = Images(args.input_dir, duplicates=args.duplicates, transform=transform)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    model = PULSE(args, device, normalize=False, verbose=True).to(device)
    for ref_im, ref_im_name in dataloader:
        if args.save_intermediate:
            padding = ceil(log10(100))
            for i in range(args.batch_size):
                int_path_HR = Path(out_path / ref_im_name[i] / "HR")
                int_path_HR.mkdir(parents=True, exist_ok=True)
            for j, (SR, LR) in enumerate(model(ref_im)):
                for i in range(args.batch_size):
                    image_SR = Image.fromarray(make_image(SR[i]))
                    image_SR.save(
                            int_path_HR / f"{ref_im_name[i]}_{j:0{padding}}.png")

        else:
            for j, (SR, LR) in enumerate(model(ref_im)):
                for i in range(args.batch_size):
                    image_SR = Image.fromarray(make_image(SR[i]))
                    image_SR.save(
                        out_path / f"{ref_im_name[i]}.png")

