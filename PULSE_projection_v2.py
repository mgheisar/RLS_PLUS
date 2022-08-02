# Everything is same as Projector except loss function
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

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

torch.cuda.set_device(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    return noises


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
        # # ------------------------------------------------------------AA
        # # ------------------------------------------------------------AA

    def forward(self, ref_im, ref_pred):
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True
        batch_size = ref_im.shape[0]
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
        n_mean_latent = 10000
        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, 512, device=device)
            latent_out = self.g_ema.style(noise_sample)

            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
        #
        if self.args.w_plus:
            latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(batch_size, 1)
            latent_in = latent_in.unsqueeze(1).repeat(1, self.g_ema.n_latent, 1)
        else:
            latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(batch_size, 1, 1)
        latent_in.requires_grad = True
        var_list = [latent_in] + noises
        # # ------------------------------------------------------------AA

        # # ------------------------------------------------------------BB**********************
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
        for j in range(self.args.steps):
            t = j / self.args.steps
            # # ------------------------------------------------------------BB*********************
            # lr = get_lr(t, self.args.learning_rate, self.args.lr_rampdown)
            # opt.param_groups[0]["lr"] = lr
            opt.zero_grad()
            # # ------------------------------------------------------------BB
            if self.args.w_plus:
                latent_n = latent_in
            else:
                latent_n = latent_in.expand(-1, self.g_ema.n_latent, -1)
            # # ------------------------------------------------------------AA
            noise_strength = latent_std * self.args.noise * max(0, 1 - t / self.args.noise_ramp) ** 2
            latent_n = latent_noise(latent_n, noise_strength.item())
            gen_im, _ = self.g_ema([latent_n], input_is_latent=True, noise=noises)
            # # ------------------------------------------------------------AA
            # Normalize image to [0,1] instead of [-1,1]
            if self.normalize:
                gen_im = normalize_image_(gen_im)
            # Calculate Losses
            fake_pred = self.discriminator(gen_im)
            loss, loss_dict = loss_builder(latent_n, gen_im, fake_pred, noises)
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
                best_im = gen_im.clone().detach()
                best_latent = latent_n.clone().detach()
                best_noises = noises

            if self.args.save_intermediate:
                yield best_im, best_latent, best_noises

            loss.backward()
            opt.step()
            scheduler.step()  # --------------*************************************************
            noise_normalize_(noises)

        total_t = time.time() - start_t
        current_info = f' | time: {total_t:.1f}'
        if self.verbose: print(best_summary + current_info)
        # import lpips
        # from torch.nn import functional as F
        # # perceptual = lpips.PerceptualLoss(
        # #     model="net-lin", net="vgg", use_gpu=self.device.startswith("cuda"),
        # #     gpu_ids=[int(self.args.gpu_num)])
        # percept_val = 0
        # # percept_val = perceptual(best_im, ref_im).sum()
        # fake_pred = self.discriminator(best_im)
        # l1_val = F.l1_loss(best_im, ref_im)
        # l2_val = F.mse_loss(best_im, ref_im)
        # adv_val = F.softplus(-fake_pred).mean()
        # print(f'L1: {l1_val:.4f}|L2:{l2_val:.4f}|percept:{percept_val:.4f}'
        #       f'|adversarial:{adv_val:.4f}')
        # best_im, _ = self.g_ema([best_latent], input_is_latent=True)
        yield best_im, best_latent, best_noises


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Projection"
    )
    # I/O arguments
    parser.add_argument('--input_dir', type=str, default='input/superres/images', help='input data directory')
    parser.add_argument('--output_dir', type=str, default='input/superres/inversion', help='output data directory')
    parser.add_argument('--duplicates', type=int, default=1,
                        help='How many HR images to produce for every image in the input directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to use during optimization')
    parser.add_argument("--size", type=int, default=128, help="output image size")

    parser.add_argument("--ckpt", type=str, default="checkpoint/BBBC021/150000.pt", help="path to the model checkpoint")
    parser.add_argument("--w_plus", action="store_false", help="allow to use distinct latent codes to each layers")
    parser.add_argument("--noise", type=float, default=0.05, help="strength of the noise level")
    parser.add_argument("--noise_ramp", type=float, default=0.75, help="duration of the noise level decay", )
    parser.add_argument('--save_intermediate', action="store_true",
                        help='Whether to store and save intermediate images during optimization')
    parser.add_argument('--num_trainable_noise_layers', type=int, default=0, help='Number of noise layers to optimize')

    parser.add_argument('--learning_rate', type=float, default=0.5, help='Learning rate to use during optimization')
    parser.add_argument('--factor', type=int, default=1, help='Super resolution factor')
    parser.add_argument('--steps', type=int, default=500, help='Number of optimization steps')
    parser.add_argument('--lr_schedule', type=str, default='linear1cycledrop',
                        help='fixed, linear1cycledrop, linear1cycle')
    parser.add_argument("--lr_rampdown", type=float, default=0.25, help="duration of the learning rate decay")
    parser.add_argument("--gpu_num", type=int, default=1, help="gpu number")
    parser.add_argument('--loss_str', type=str, default="1*L1+0.1*Percept+0.1*Adv",
                        help='Loss function to use')  # "1*L2+0.05*Adv"

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_num)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    files = sorted(glob.glob(os.path.join(args.input_dir, '*.png')))
    input_dir = args.input_dir + '/image_128/'
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
        for i in range(len(files)):
            img = Image.open(files[i])
            width, height = img.size  # Get dimensions
            left = (width - args.size) / 2
            top = (height - args.size) / 2
            right = (width + args.size) / 2
            bottom = (height + args.size) / 2
            img = img.crop((left, top, right, bottom))
            img.save(f'{input_dir}{files[i].split("/")[-1]}')

    dataset = Images(input_dir, duplicates=args.duplicates, transform=transform)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    model = PULSE(args, device, normalize=False, verbose=True).to(device)
    latents, noises = [], []
    # loss_vec = ["1*L2+0.1*Adv", "1*L2", "1*L2+0.05*Adv", "1*L1", "1*L1+0.05*Adv", "1*L1+0.1*Adv"]
    # for ii in range(len(loss_vec)):
    # args.loss_str = loss_vec[ii]
    for ref_im, ref_im_name in dataloader:
        ref_pred = model.discriminator(ref_im).detach()
        for j, (image, latent, noise) in enumerate(model(ref_im, ref_pred)):
            for i in range(args.batch_size):
                img_ar = make_image(image[i])
                pil_img = Image.fromarray(img_ar)
                pil_img.save(
                    out_path / f"{ref_im_name[i]}_{args.loss_str}-{args.learning_rate}.png")
                latents.append(latent)
                noises.append(noise)
