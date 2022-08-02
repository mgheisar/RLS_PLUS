import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import IncrementalPCA
import torch
import argparse
import CPFLOW_lib.datasets as datasets
from pingouin import multivariate_normality
from CPFLOW_lib.flows import (SequentialFlow, DeepConvexFlow, ActNorm)
from CPFLOW_lib.utils_nf import batch_iter, load_arch, seed_prng
import CPFLOW_lib.utils as utils
import os

torch.cuda.set_device(0)
cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def cvt(x):
    return x.to(device=device, dtype=dtype, memory_format=torch.contiguous_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--clip-grad', type=float, default=0)
    parser.add_argument(
        '--preload-data', action='store_true', default=False,
        help="Preload entire dataset to GPU (if cuda).")
    parser.add_argument('--save', help='directory to save results', type=str,
                        default='/projects/superres/Marzieh/CP-Flow/experiments/cpfw_512_2')
    parser.add_argument('--dimh', type=int, default=4096)  # 64:img
    parser.add_argument('--nhidden', type=int, default=5)  # 4:img
    parser.add_argument("--nblocks", type=int, default=10, help='Number of stacked CPFs.')  # 8-8-8 img
    parser.add_argument('--atol', type=float, default=1e-3)
    parser.add_argument('--rtol', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--val_batch_size', type=int, default=1024)
    parser.add_argument('--fp64', action='store_true', default=False)
    parser.add_argument('--brute_val', action='store_true', default=False)

    args = parser.parse_args()
    args.test_batch_size = args.batch_size
    seed_prng(args.seed, cuda=cuda)
    if args.fp64:
        torch.set_default_dtype(torch.float64)
    dtype = torch.float32 if not args.fp64 else torch.float64
    data = datasets.WLATENT(datasets.root)
    data_x = torch.from_numpy(data.trn.x[:5000, :])
    w_samples = data.trn_n.x[:5000, :]
    p_samples = torch.nn.LeakyReLU(5)(torch.from_numpy(w_samples))
    p_samples = p_samples.numpy()

    n_components = 512
    transformer = IncrementalPCA(n_components, whiten=False, batch_size=max(100, 5 * n_components))
    X_mean = p_samples.mean(0)
    transformer.fit(p_samples - p_samples.mean(0))
    X_comp = transformer.components_
    X_stdev = np.sqrt(transformer.explained_variance_)  # already sorted
    # var_ratio = transformer.explained_variance_ratio_
    pn_samples = np.matmul((p_samples - X_mean), X_comp.T) / X_stdev

    Arch = load_arch(args.arch)
    icnns = [Arch(data.n_dims, args.dimh, args.nhidden,
                  softplus_type=args.softplus_type,
                  zero_softplus=args.zero_softplus,
                  symm_act_first=args.symm_act_first) for _ in range(args.nblocks)]
    layers = [None] * (2 * args.nblocks + 1)
    layers[0::2] = [ActNorm(data.n_dims) for _ in range(args.nblocks + 1)]
    layers[1::2] = [DeepConvexFlow(icnn, data.n_dims, unbiased=False,
                                   atol=args.atol, rtol=args.rtol,
                                   trainable_w0=args.trainable_w0) for _, icnn in zip(range(args.nblocks), icnns)]

    flow = SequentialFlow(layers)
    flow = flow.to(device=device, dtype=dtype)
    # deal with data-dependent initialization like actnorm.
    with torch.no_grad():
        x = torch.rand(8, data.n_dims).to(device)
        flow.forward_transform(x)
    map_location = lambda storage, loc: storage.cuda()
    most_recent_path = torch.load(os.path.join(args.save, 'best_model.pth'), map_location=map_location)

    flow.load_state_dict(most_recent_path['model'])
    with torch.no_grad():
        flow.eval()
        for f in flow.flows[1::2]:
            f.no_bruteforce = not args.brute_val
            debug = False
            no_grad = True
        torch.autograd.set_detect_anomaly(debug)
        with torch.set_grad_enabled(mode=not no_grad):
            nf_samples = []
            val_logp = utils.AverageMeter()
            for x in batch_iter(data_x, batch_size=args.val_batch_size):  # evalD.x
                x = cvt(x)
                val_logp.update(flow.logp(x).mean().item(), x.size(0))
                z = flow.forward_transform(x)[0]
                nf_samples.append(z.data.cpu().numpy())

    print("log probability: ", val_logp.avg)
    nf_samples = np.vstack(nf_samples)
    # np.save('nf_samples_2.npy', nf_samples)
    # nf_samples = np.load('nf_samples_2.npy')

    subset = {"w_space": w_samples, "p_space": p_samples,
              "pn_space": pn_samples, "nf_space": nf_samples}
    spaces = ['w_space', 'p_space', 'pn_space', 'nf_space']
    for space in spaces:
        measurements = subset[space]
        a = np.power(np.linalg.norm(measurements, axis=1), 2)
        df = pd.DataFrame({"x": a})
        sns.distplot(df, hist=True, kde=True, label=space)
    xx = np.arange(0, 1000, 0.1)
    plt.plot(xx, stats.chi2.pdf(xx, df=512), label="X2(512)")
    plt.legend()
    plt.xlabel('Squared L2 norm')
    # plt.savefig('chi-squared')
    plt.show()
    ind = np.random.choice(range(512), 3, replace=False)
    print(ind)
    tt = []
    for space in spaces:
        measurements = subset[space]
        x1 = measurements[:, 505]
        x2 = measurements[:, 506]
        x3 = measurements[:, 508]
        df = pd.DataFrame({"x": x1, "y": x2, "z": x3})
        pd.plotting.scatter_matrix(df, diagonal='kde')
        plt.suptitle(space)
        plt.show()
        print(f'shapiro x1:{stats.shapiro(x1)[1]}')
        print(f'shapiro x2:{stats.shapiro(x2)[1]}')
        print(f'shapiro x3:{stats.shapiro(x3)[1]}')
        HZResults = multivariate_normality(pd.DataFrame(measurements), alpha=.05)
        p = np.zeros((data.n_dims,))
        for ii in range(data.n_dims):
            x1 = measurements[:, ii]
            stat, p[ii] = stats.shapiro(x1)
        p_val = np.min(p)
        print(f'{space} shapiro stat, hz stat: ', np.min(p), HZResults.pval)
        print('number of shapiro stat > 0.05: ', len(np.where(p > 0.05)[0]))
        tt.append(np.where(p > 0.05)[0])

    # print('\n -----------------------------------------------------------------')
    # for i, x in enumerate(tt[3]):
    #     if len(np.where(tt[2] == x)[0]) == 0:
    #         if len(np.where(tt[1] == x)[0]) == 0:
    #             if len(np.where(tt[0] == x)[0]) == 0:
    #                 print(x)

