import argparse

import torch
from torchvision import utils

from model import Generator
# Face: -i 19 -d 5 -n 5 --ckpt "checkpoint/stylegan2-ffhq-config-f.pt" --size 1024 factor_face.pt --out_prefix
# face_factor --truncation=0.5
torch.cuda.set_device(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1)
if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-i", "--index", type=int, default=4, help="index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=4,
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
        "--truncation", type=float, default=1, help="truncation factor"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="factor",
        help="filename prefix to result samples",
    )
    parser.add_argument(
        "--factor",
        type=str,
        default="factor.pt",
        help="name of the closed form factorization result factor file",
    )

    args = parser.parse_args()
    map_location = lambda storage, loc: storage.cuda()
    eigvec = torch.load(args.factor, map_location=map_location)["eigvec"].to(device)
    ckpt = torch.load(args.ckpt, map_location=map_location)
    # g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier).to(device)
    g = Generator(args.size, 512, 8).to(device)
    g.load_state_dict(ckpt["g_ema"], strict=False)

    # trunc = g.mean_latent(4096)

    # latent = torch.randn(args.n_sample, 512, device=device)
    # latent = g.get_latent(latent)
    latent = torch.load('w_nf', map_location=map_location)
    noises = torch.load('w_nf_n', map_location=map_location)

    direction = args.degree * eigvec[:, args.index].unsqueeze(0)
    # img, _ = g(
    #     [latent],
    #     truncation=args.truncation,
    #     truncation_latent=trunc,
    #     input_is_latent=True,
    # )
    print('degree:',  args.degree)
    img, _ = g(
        [latent],
        input_is_latent=True,
        noise=noises
    )
    img1, _ = g(
        [latent + direction],
        input_is_latent=True,
        noise=noises
    )
    img2, _ = g(
        [latent - direction],
        input_is_latent=True,
        noise=noises
    )
    A = torch.cat([img1, img, img2], 0)
    grid = utils.save_image(
        torch.cat([img1, img, img2], 0),
        f"{args.out_prefix}_index_trunc-{args.truncation}-{args.index}_degree-{args.degree}1_.png",
        normalize=True,
        range=(-1, 1),
        nrow=args.n_sample,
    )