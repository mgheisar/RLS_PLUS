import argparse
import numpy as np
import torch
from torchvision import utils

from model import Generator
for seed in range(4, 5):
    torch.manual_seed(seed)
    torch.cuda.set_device(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if __name__ == "__main__":
        torch.set_grad_enabled(False)

        parser = argparse.ArgumentParser(description="Apply closed form factorization")
        parser.add_argument("--ckpt", type=str, default="checkpoint/BBBC021/150000.pt", help="stylegan2 checkpoints")
        parser.add_argument(
            "--size", type=int, default=128, help="output image size of the generator"
        )
        parser.add_argument(
            "--out_prefix",
            type=str,
            default="factor",
            help="filename prefix to result samples",
        )
        parser.add_argument(
            "--factor", default="factor_BBBC.pt",
            type=str,
            help="name of the closed form factorization result factor file",
        )

        args = parser.parse_args()
        num_direction = 25
        eigvec = torch.load(args.factor)["eigvec"].to(device)
        map_location = lambda storage, loc: storage.cuda()
        ckpt = torch.load(args.ckpt, map_location=map_location)
        g = Generator(args.size, 512, 8).to(device)
        g.load_state_dict(ckpt["g_ema"], strict=False)

        latent = torch.load('w_i2s2', map_location=map_location)
        latent2 = torch.load('w_i2s2', map_location=map_location)
        noises = torch.load('w_i2s2_n', map_location=map_location)
        # latent = torch.randn(1, 512, device=device)
        # latent = g.get_latent(latent)
        # noises_single = g.make_noise()
        # noises = []
        # for noise in noises_single:
        #     noises.append(noise.repeat(1, 1, 1, 1).normal_())
        # alphaa = torch.linspace(0, 1, steps=10)
        # imgs = []
        # for index in range(len(alphaa)):
        #     a = (1-alphaa[index])
        #     b = a * latent
        #     c = alphaa[index] * latent2
        #     img, _ = g(
        #         [(1-alphaa[index]) * latent + alphaa[index] * latent2],
        #         input_is_latent=True,
        #         noise=noises
        #     )
        #     imgs.append(img)
        # imgs = torch.stack(imgs)
        # imgs = imgs.view(-1, *imgs.shape[2:])
        # grid = utils.save_image(
        #     imgs,
        #     f"apply_factor_cells/temp/testnf.png",
        #     normalize=True,
        #     range=(-1, 1),
        #     nrow=1,
        # )
        # exit(0)

        img, _ = g(
            [latent],
            input_is_latent=True,
            noise=noises
        )
        imgs = []
        dd = np.arange(1, 10, 2)
        for index in range(10, num_direction):
            imgs_plus, imgs_minus = [], []
            for degree in dd:
                direction = degree * eigvec[:, index].unsqueeze(0)
                img1, _ = g(
                    [latent + direction],
                    input_is_latent=True,
                    noise=noises
                )
                imgs_plus.append(img1)
                img2, _ = g(
                    [latent - direction],
                    input_is_latent=True,
                    noise=noises
                )
                imgs_minus.append(img2)

            imgs_minus = torch.stack(imgs_minus)
            imgs_minus = imgs_minus.view(-1, *imgs_minus.shape[2:])

            imgs_plus = torch.stack(imgs_plus)
            imgs_plus = imgs_plus.view(-1, *imgs_plus.shape[2:])
            A = torch.cat([imgs_minus, img, imgs_plus], 0)
            imgs.append(A)

        imgs = torch.stack(imgs)
        imgs = imgs.view(-1, *imgs.shape[2:])
        grid = utils.save_image(
            imgs,
            f"apply_factor_cells/temp/{args.out_prefix}_i2s2.png",
            normalize=True,
            range=(-1, 1),
            nrow=2*len(dd)+1,
        )