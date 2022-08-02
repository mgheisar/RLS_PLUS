import argparse
import numpy as np
import torch
from torchvision import utils
from model import Generator

# Domains = ['DMSO_656', 'cytochalasin B_0.01', 'cytochalasin D_0.003', 'demecolcine_0.003']
# Domains = ['DMSO_656', 'demecolcine_0.003', 'demecolcine_0.01', 'demecolcine_0.03',
#            'demecolcine_10.0', 'cytochalasin B_0.01', 'cytochalasin B_0.1',
#            'cytochalasin B_0.3', 'cytochalasin B_1.0', 'cytochalasin B_3.0',
#            'cytochalasin B_10.0', 'cytochalasin D_0.003', 'cytochalasin D_0.01',
#            'cytochalasin D_0.3', 'cytochalasin D_3.0', 'staurosporine_0.003']
torch.cuda.set_device(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description="Domain Translation")
    parser.add_argument("--channel_multiplier", type=int, default=2,
                        help='channel multiplier factor. config-f = 2, else = 1', )
    parser.add_argument("--ckpt", type=str, default="checkpoint/BBBC021/150000.pt",
                        help="stylegan2 checkpoints")
    parser.add_argument("--size", type=int, default=128, help="output image size of the generator")
    parser.add_argument("-n", "--n_sample", type=int, default=30, help="number of samples created")
    parser.add_argument("--device", type=str, default="cuda", help="device to run the model")
    parser.add_argument("--ds", type=str, default="DMSO_656", help="name of the source domain")

    parser.add_argument("--dt", type=str, default="demecolcine_10.0", help="name of the target domain")
    parser.add_argument("--truncation", type=float, default=0.5, help="truncation factor")
    parser.add_argument("--degree", type=float, default=1, help="scalar factors for moving latent vectors along mean")

    args = parser.parse_args()
    map_location = lambda storage, loc: storage.cuda()
    Loss_vec = ["1*L1", "1*L1+0.1*Percept", "1*L1+0.1*Percept+0.1*Adv", "1*L1+1*Percept+0.1*Adv"]
    # args.loss_str = Loss_vec[2]
    # latent_dir = "Domain_Projection_Prev/500Steps_/"

    args.loss_str = Loss_vec[3]
    latent_dir = "Domain_Projection_Prev/1000Steps_without_noise/"

    ds_files = torch.load(latent_dir + args.ds + args.loss_str + ".pt", map_location=map_location)
    dt_files = torch.load(latent_dir + args.dt + args.loss_str + ".pt", map_location=map_location)

    g = Generator(args.size, 512, 8).to(device)
    g.load_state_dict(torch.load(args.ckpt, map_location=map_location)["g_ema"], strict=False)
    g.eval()
    noises_single = g.make_noise()
    ds_noises, dt_noises, noises = [], [], []
    for noise in noises_single:
        ds_noises.append(noise.repeat(args.n_sample, 1, 1, 1).normal_())
        dt_noises.append(noise.repeat(args.n_sample, 1, 1, 1).normal_())
        noises.append(noise.repeat(args.n_sample, 1, 1, 1).normal_())

    # trunc = g.mean_latent(10000)
    ds_list = list(ds_files.keys())
    dt_list = list(dt_files.keys())
    ds_ind = torch.randint(high=len(ds_list), size=(args.n_sample,), device=device)
    dt_ind = torch.randint(high=len(dt_list), size=(args.n_sample,), device=device)
    ds_latent, dt_latent = [], []
    for i in range(args.n_sample):
        key = ds_list[ds_ind[i]]
        ds_latent.append(ds_files[key]['latent'])
        n_i = 0
        for noise in ds_files[key]['noise']:
            ds_noises[n_i][i, :] = noise[0]
            n_i += 1

        key = dt_list[dt_ind[i]]
        dt_latent.append(dt_files[key]['latent'])
        n_i = 0
        for noise in dt_files[key]['noise']:
            dt_noises[n_i][i, :] = noise[0]
            n_i += 1

    ds_latent = torch.stack(ds_latent)
    ds_mean = torch.mean(ds_latent, dim=0)
    dt_latent = torch.stack(dt_latent)
    dt_mean = torch.mean(dt_latent, dim=0)
    # ds_noise_mean, dt_noise_mean = [], []
    # for noise in ds_noises:
    #     ds_noise_mean.append(torch.mean(noise, dim=0).repeat(args.n_sample, 1, 1, 1))
    # for noise in dt_noises:
    #     dt_noise_mean.append(torch.mean(noise, dim=0).repeat(args.n_sample, 1, 1, 1))

    ds_dt = ds_mean - dt_mean
    dt_ds = dt_mean - ds_mean
    img_s, _ = g(
        [ds_latent],
        input_is_latent=True,
        noise=ds_noises
    )

    img_t, _ = g(
        [dt_latent],
        input_is_latent=True,
        noise=dt_noises
    )
    dt_ds = dt_latent - ds_latent
    # dt_ds = torch.nn.functional.normalize(dt_latent - ds_latent, p=2.0, dim=1)
    img_stt = []
    for dd in np.arange(0, 1.1, 0.1):
        args.degree = dd
        img_st, _ = g(
            [ds_latent + args.degree * dt_ds],
            input_is_latent=True,
            noise=noises
        )
        img_stt.append(img_st)
    img_st = torch.stack(img_stt)
    img_st = img_st.view(-1, *img_st.shape[2:])

    # dt_ds = dt_mean - ds_mean
    # img_st, _ = g(
    #     [ds_latent + args.degree * dt_ds],
    #     input_is_latent=True,
    #     noise=ds_noises
    # )

    grid = utils.save_image(
        torch.cat([img_s, img_st, img_t], 0),
        f"{args.ds}_{args.dt}_st_500.png",
        normalize=True,
        range=(-1, 1),
        nrow=args.n_sample,
    )

    # # trunc_s = g.mean_latent(10000)
    # # trunc_s = ds_mean
    # trunc_t = dt_mean  # <---------------------------------------
    # img_t = []
    # for tc in np.arange(0, 1.1, 0.1):
    #     img, _ = g(
    #         [dt_latent],
    #         input_is_latent=True,
    #         truncation=tc,
    #         truncation_latent=trunc_t,
    #         noise=dt_noises
    #     )
    #     img_t.append(img)
    #
    # grid = utils.save_image(torch.cat(img_t, 0), f"{args.dt}+.png",
    #                         normalize=True, range=(-1, 1), nrow=args.n_sample,)
