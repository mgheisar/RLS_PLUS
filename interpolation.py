import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from model import Generator
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
torch.manual_seed(1)

# Face -i 10 --ckpt "checkpoint/stylegan2-ffhq-config-f.pt" "factor_face.pt" --size 1024 --out_prefix face_factor
# DMSO -i 10 --ckpt "checkpoint/100000.pt" "factor.pt" --size 128 --out_prefix factor --truncation=0.4

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-i", "--index", type=int, default=0, help="index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument("--ckpt", type=str, default="checkpoint/BBBC021/150000.pt", help="stylegan2 checkpoints")
    parser.add_argument(
        "--size", type=int, default=128, help="output image size of the generator"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=1, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.6, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eigvec = torch.load(args.factor, map_location=device)["eigvec"].to(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier).to(args.device)
    g.load_state_dict(ckpt["g_ema"], strict=False)

    trunc = g.mean_latent(4096)
    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(latent)

    dd_vec = np.arange(10)
    for i in range(25):
        args.index = i
        fig = plt.figure()  # make figure
        ims = []
        for dd in dd_vec:
            args.degree = dd
            direction = args.degree * eigvec[:, args.index].unsqueeze(0)

            # img, _ = g(
            #     [latent],
            #     truncation=args.truncation,
            #     truncation_latent=trunc,
            #     input_is_latent=True,
            # )
            img1, _ = g(
                [latent + direction],
                truncation=args.truncation,
                truncation_latent=trunc,
                input_is_latent=True,
            )
            # img2, _ = g(
            #     [latent - direction],
            #     truncation=args.truncation,
            #     truncation_latent=trunc,
            #     input_is_latent=True,
            # )

            imag = img1.clamp_(-1, 1).add_(1).div_(2.0)
            ims.append(imag.squeeze(0).cpu().permute(1, 2, 0).numpy())

        im = plt.imshow(ims[0])


        # function to update figure
        def updatefig(j):
            # set the data in the axesimage object
            im.set_array(ims[j])
            # return the artists set
            return [im]


        # kick off the animation
        anim = animation.FuncAnimation(fig, updatefig, frames=range(len(dd_vec)), interval=50, blit=True)
        anim.save('gifs/animation_%s_index_%d_trunc%f.gif' % (args.out_prefix, args.index, args.truncation), writer='imagemagick', fps=4)
        # plt.show()
