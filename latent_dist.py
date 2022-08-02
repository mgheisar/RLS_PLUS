import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import IncrementalPCA
import argparse
from model import Generator
from FLOWS import flows as fnn
import pickle
import os
from pingouin import multivariate_normality


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

torch.cuda.set_device(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
                        default='/projects/superres/Marzieh/pytorch-flows/experiments/maf_lr_5')
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
    size = 128
    ckpt = "checkpoint/BBBC021/150000.pt"
    g_ema = Generator(size, 512, 8).to(device)
    map_location = lambda storage, loc: storage.cuda()
    gan = torch.load(ckpt, map_location=map_location)
    g_ema.load_state_dict(gan["g_ema"], strict=False)
    g_ema.eval()
    del gan
    torch.cuda.empty_cache()
    # # # -------Loading NF model----------------------------------------------------------------------------
    num_blocks = 5
    num_inputs = 512
    modules = []
    for _ in range(num_blocks):
        modules += [
            fnn.MADE(num_inputs, num_hidden=1024, num_cond_inputs=None, act='relu'),
            fnn.BatchNormFlow(num_inputs),
            fnn.Reverse(num_inputs)
        ]
    flow = fnn.FlowSequential(*modules)
    map_location = lambda storage, loc: storage.cuda()
    best_model_path = torch.load(os.path.join(args.save, 'best_model.pth'), map_location=map_location)
    flow.load_state_dict(best_model_path['model'])
    flow.to(device)
    with open('wlatent1.pkl', 'rb') as f:
        data = pickle.load(f)
    flow.eval()
    for param in flow.parameters():
        param.requires_grad = False
    # # ------------------------------------------------------------AA
    lrelu = torch.nn.LeakyReLU(negative_slope=0.2)
    with torch.no_grad():
        z_samples = torch.randn((5000, 512), dtype=torch.float32, device=device)
        w_samples = g_ema.style(z_samples)
    p_samples = torch.nn.LeakyReLU(5)(w_samples)
    p_samples = p_samples.cpu().numpy()

    n_components = 512
    transformer = IncrementalPCA(n_components, whiten=False, batch_size=max(100, 5 * n_components))
    X_mean = p_samples.mean(0)
    transformer.fit(p_samples - p_samples.mean(0))
    X_comp = transformer.components_
    X_stdev = np.sqrt(transformer.explained_variance_)  # already sorted
    # var_ratio = transformer.explained_variance_ratio_
    pn_samples = np.matmul((p_samples - X_mean), X_comp.T) / X_stdev
    nf_samples = []
    logp_ = []
    train_loader = torch.utils.data.DataLoader(
        w_samples, batch_size=args.val_batch_size, shuffle=True)
    for batch_id, x in enumerate(train_loader):  # evalD.x
        x = torch.nn.LeakyReLU(5)(x)
        x = (x - torch.from_numpy(data["mu"]).to(x)) / torch.from_numpy(data["std"]).to(x)
        logp = flow.log_probs(x, None).mean()
        z = flow.forward(x, None, mode='direct')[0]
        x_rec = flow.forward(z, None, mode='inverse')[0]
        nf_samples.append(z.data.cpu().numpy())
        logp_.append(logp.cpu().numpy())
    nf_samples = np.vstack(nf_samples)
    print("log probability: ", np.mean(np.vstack(logp_)))

    np.save('nf_samples_2.npy', nf_samples)
    nf_samples = np.load('nf_samples_2.npy')
    w_samples = w_samples.cpu().numpy()
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
    plt.savefig('chi-squared')
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
        p = np.zeros((512,))
        for ii in range(512):
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

