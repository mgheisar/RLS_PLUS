# Everything is same as Projector I2S
from model import Generator, Discriminator
from SphericalOptimizer import SphericalOptimizer
import numpy as np
import time
from loss_projection import LossBuilder
from functools import partial
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch
from PIL import Image
import argparse
from torchvision import transforms
import math
from tqdm import tqdm
import ii2s_net

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True

torch.cuda.set_device(0)
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
            .permute(1, 2, 0)
            .to("cpu")
            .numpy()
    )


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
        if (self.duplicates == 1):
            return image, img_path.stem
        else:
            return image, img_path.stem + f"_{(idx % self.duplicates) + 1}"


class PULSE(torch.nn.Module):
    def __init__(self, args, device, verbose=True):
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
        # # ------------------------------------------------------------AA
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)
        if self.verbose: print("\tRunning Mapping Network")
        with torch.no_grad():
            latent = torch.randn((1000000, 512), dtype=torch.float32, device=self.device)
            # latent_out = torch.nn.LeakyReLU(5)(self.g_ema.style(latent))
            latent_out = self.g_ema.style(latent)
            self.gaussian_fit = {"mean": latent_out.mean(0), "std": latent_out.std(0)}
        # # ------------------------------------------------------------AA

    def forward(self, ref_im):
        batch_size = ref_im.shape[0]
        # # ------------------------------------------------------------AA
        noises = []  # stores all of the noise tensors
        noise_vars = []  # stores the noise tensors that we want to optimize on
        for i in range(self.g_ema.n_latent - 1):
            # dimension of the ith noise tensor
            res = (batch_size, 1, 2 ** ((i + 1) // 2 + 2), 2 ** ((i + 1) // 2 + 2))
            new_noise = torch.randn(res, dtype=torch.float, device=self.device)
            if i < self.args.num_trainable_noise_layers:
                new_noise.requires_grad = True
                noise_vars.append(new_noise)
            else:
                new_noise.requires_grad = False

            noises.append(new_noise)

        # Generate latent tensor
        if self.args.w_plus:
            # latent = torch.randn(
            #     (batch_size, self.g_ema.n_latent, 512), dtype=torch.float, requires_grad=True, device=self.device)
            latent = self.gaussian_fit["mean"].detach().clone().unsqueeze(0).repeat(batch_size, 1)
            latent = latent.unsqueeze(1).repeat(1, self.g_ema.n_latent, 1)
            latent.requires_grad = True
        else:
            # latent = torch.randn(
            #     (batch_size, 512), dtype=torch.float, requires_grad=True, device=self.device)
            latent = self.gaussian_fit["mean"].detach().clone().repeat(batch_size, 1)
            latent.requires_grad = True
        var_list = [latent] + noise_vars
        # # ------------------------------------------------------------AA

        # # ------------------------------------------------------------BB
        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        # opt_func = opt_dict[self.args.opt_name]
        # opt = SphericalOptimizer(opt_func, var_list, lr=self.args.learning_rate)
        pca = ii2s_net.Net(args)
        opt = torch.optim.Adam([latent] + noise_vars, lr=args.learning_rate)
        schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9 * (1 - np.abs(x / self.args.steps - 1 / 2) * 2) + 1) / 10,
            'linear1cycledrop': lambda x: (9 * (
                    1 - np.abs(
                x / (0.9 * self.args.steps) - 1 / 2) * 2) + 1) / 10 if x < 0.9 * self.args.steps else 1 / 10 + (
                    x - 0.9 * self.args.steps) / (0.1 * self.args.steps) * (1 / 1000 - 1 / 10),
        }
        schedule_func = schedule_dict[self.args.lr_schedule]
        # scheduler = torch.optim.lr_scheduler.LambdaLR(opt.opt, schedule_func)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule_func)
        # # -------------------------------------------------------------BB
        loss_builder = LossBuilder(ref_im, None, self.args.loss_str,
                                   device=self.device, gpu_num=self.args.gpu_num)

        min_loss = np.inf
        best_summary = ""
        start_t = time.time()
        gen_im = None

        if self.verbose:
            print("Optimizing")
        pbar = tqdm(range(args.steps))
        for j in pbar:
            t = j / self.args.steps
            # # ------------------------------------------------------------BB
            # opt.opt.zero_grad()
            opt.zero_grad()
            # # ------------------------------------------------------------BB

            # # ------------------------------------------------------------AA
            latent_in = latent
            # # Apply learned linear mapping to match latent distribution to that of the mapping network
            # latent_in = self.lrelu(latent_in * self.gaussian_fit["std"] + self.gaussian_fit["mean"])
            # latent_in = self.lrelu(latent_in)
            gen_im, _ = self.g_ema([latent_in], input_is_latent=True, noise=noises)
            # # ------------------------------------------------------------AA
            # Normalize image to [0,1] instead of [-1,1]
            # gen_im = (gen_im + 1) / 2

            # Calculate Losses
            # fake_pred = self.discriminator(gen_im)
            fake_pred = None
            loss, loss_dict = loss_builder(latent_in, gen_im, fake_pred, noises)
            p_norm_loss = pca.cal_p_norm_loss(latent_in)
            loss += args.p_norm_lambda * p_norm_loss
            loss_dict['p_norm'] = p_norm_loss
            loss_dict['TOTAL'] = loss

            loss.backward()
            opt.step()
            # # ------------------------------------------------------------BB
            scheduler.step()
            # # ------------------------------------------------------------BB
            noise_normalize_(noises)
            # Save best summary for log
            if loss < min_loss:
                min_loss = loss
                best_summary = f'BEST ({j + 1}) | ' + ' | '.join(
                    [f'{x}: {y:.4f}' for x, y in loss_dict.items()])
                best_im = gen_im.detach().clone()
                best_latent = latent_in.detach().clone()
                best_noises = noises

            pbar.set_description(
                (
                    f"Percept: {loss_dict['Percept']:.4f};"
                    f" L1: {loss_dict['L1']:.4f};"
                    f"P_norm_loss:{p_norm_loss:.4f};"
                    f" Geocross: {loss_dict['GEOCROSS']:.4f};|"
                    f" TOTAL: {loss:.4f},"

                )
            )
            # Save intermediate HR and LR images
            if self.args.save_intermediate:
                yield gen_im, latent_in, noises

        total_t = time.time() - start_t
        current_info = f' | time: {total_t:.1f} | it/s: {(j + 1) / total_t:.2f} | batchsize: {batch_size}'
        if self.verbose: print(best_summary + current_info)
        yield best_im, best_latent, best_noises


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Projection with PULSE"
    )
    # I/O arguments
    parser.add_argument('--input_dir', type=str, default='input', help='input data directory')
    parser.add_argument('--output_dir', type=str, default='input/project', help='output data directory')
    parser.add_argument('--duplicates', type=int, default=1,
                        help='How many HR images to produce for every image in the input directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to use during optimization')
    parser.add_argument("--size", type=int, default=128, help="output image size")
    # PULSE arguments
    parser.add_argument("--ckpt", type=str, default="checkpoint/BBBC021/150000.pt", help="path to the model checkpoint")
    parser.add_argument('--exp', type=str, default='comp', help='exp name')

    parser.add_argument("--w_plus", action="store_true", help="allow to use distinct latent codes to each layers")
    parser.add_argument('--loss_str', type=str, default="1*L1+0.1*Percept+0*GEOCROSS",
                        help='Loss function to use')
    parser.add_argument(
        "--p_norm_lambda", type=float, default=1e-6, help="weight of the whitening regularization")
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='Learning rate to use during optimization')  # lr=1
    parser.add_argument('--steps', type=int, default=2000, help='Number of optimization steps')
    parser.add_argument('--num_trainable_noise_layers', type=int, default=0, help='Number of noise layers to optimize')

    parser.add_argument('--eps', type=float, default=1e-3, help='Target for downscaling loss (L2)')
    parser.add_argument('--opt_name', type=str, default='adam', help='Optimizer to use in projected gradient descent')
    parser.add_argument('--lr_schedule', type=str, default='linear1cycledrop',
                        help='fixed, linear1cycledrop, linear1cycle')
    parser.add_argument('--save_intermediate', action="store_true",
                        help='Whether to store and save intermediate images during optimization')
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--gpu_num", type=int, default=0, help="gpu number")

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_num)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose(
        [
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = Images(args.input_dir, duplicates=args.duplicates, transform=transform)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    model = PULSE(args, device).to(device)
    toPIL = transforms.ToPILImage()
    latents, noises = [], []
    for ref_im, ref_im_name in dataloader:
        for j, (image, latent, noise) in enumerate(model(ref_im)):
            for i in range(args.batch_size):
                img_ar = make_image(image[i])
                pil_img = Image.fromarray(img_ar)
                torch.save(latent, "w_ii2s")
                pil_img.save(
                    out_path / f"{ref_im_name[i]}-project_w{args.w_plus}_n{args.num_trainable_noise_layers}_.png")
                latents.append(latent)
                noises.append(noise)
