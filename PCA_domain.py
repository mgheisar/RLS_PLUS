import argparse
import torch
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# --exp "3" --ds "DMSO" --dt "cytochalasin D_10.0"
# cytochalasin B_10.0  demecolcine_10.0 staurosporine_0.01 cytochalasin D_10.0
# demecolcine_10.0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description="Computing PCA on latent vectors of DS and DT")
    parser.add_argument("--ds", type=str, help="name of the source domain")
    parser.add_argument("--dt", type=str, help="name of the target domain")
    parser.add_argument('--exp', type=str, default='3', help='experiment name')
    args = parser.parse_args()
    map_location = lambda storage, loc: storage.cpu()
    Domains = ['DMSO_656', 'cytochalasin B_0.01', 'cytochalasin D_0.003', 'demecolcine_0.003']
    args.ds = Domains[0]
    args.dt = Domains[3]
    Loss_vec = ["1*L1", "1*L1+0.1*Percept", "1*L1+0.1*Percept+0.1*Adv", "1*L1+0.01*Adv+0.1*Percept"]
    args.loss_str = Loss_vec[2]
    ds_files = torch.load("Domain_Projection_Prev/500Steps/" + args.ds + args.loss_str + "_500.pt", map_location=map_location)
    # dt_files = torch.load("Domain_Projection_Prev/500Steps/" + args.dt + args.loss_str + "_500.pt", map_location=map_location)
    #
    # ds_list = list(ds_files.keys())
    # dt_list = list(dt_files.keys())
    # dst_latent, ds_latent, dt_latent = [], [], []
    # for i in range(len(dt_list)):
    #     key = ds_list[i]
    #     dst_latent.append(ds_files[key]['latent'].view(-1))
    #     ds_latent.append(ds_files[key]['latent'].view(-1))
    #     # dst_latent.append(torch.mean(ds_files[key]['latent'], dim=0))
    #     # ds_latent.append(torch.mean(ds_files[key]['latent'], dim=0))
    #
    #     key = dt_list[i]
    #     dst_latent.append(dt_files[key]['latent'].view(-1))
    #     dt_latent.append(dt_files[key]['latent'].view(-1))
    #     # dst_latent.append(torch.mean(dt_files[key]['latent'], dim=0))
    #     # dt_latent.append(torch.mean(dt_files[key]['latent'], dim=0))
    #
    # ds_latent = torch.stack(ds_latent)
    # ds_mean = torch.mean(ds_latent, dim=0)
    # dt_latent = torch.stack(dt_latent)
    # dt_mean = torch.mean(dt_latent, dim=0)
    # mean_dist = np.linalg.norm(ds_mean - dt_mean, ord=2)
    #
    # dst_latent = torch.stack(dst_latent)
    # pca = PCA(n_components=2)
    # pca.fit(dst_latent)
    # ds_points = pca.transform(ds_latent)
    # dt_points = pca.transform(dt_latent)
    # plt.subplots()
    # plt.scatter(ds_points[:, 0], ds_points[:, 1], c='r', label=args.ds)
    # plt.scatter(dt_points[:, 0], dt_points[:, 1], c='b', label=args.dt)
    # plt.legend()
    # plt.title(f'{args.loss_str}, mean_dist={mean_dist}')
    # plt.savefig(f'{args.dt}-{args.loss_str}_merge_500.png')
    # plt.show()
    #
    # pca = PCA(n_components=16)
    # pca.fit(dst_latent)
    # exp_var_pca = pca.explained_variance_ratio_
    # cum_sum_eigenvalues = np.cumsum(exp_var_pca)  # cumulative sum of variance explained with [n] features
    # # Plot the explained variance against cumulative explained variance
    # plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center',
    #         label='Individual explained variance')
    # plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
    #          label='Cumulative explained variance')
    # plt.ylabel("Explained variance ratio")
    # plt.xlabel(f"PC index {args.dt}, loss={args.loss_str}")
    # plt.legend(loc='best')
    # plt.tight_layout()
    # plt.savefig(f"Variance_ratio_{args.dt}, loss={args.loss_str}_merge_500.png")
    # plt.show()

# # -------------------------------------------------------------
# All domains together
args.loss_str = Loss_vec[1]
dt_files1 = torch.load("Domain_Projection_Prev/25Steps/" + Domains[1] + args.loss_str + "_25.pt", map_location=map_location)
dt_files2 = torch.load("Domain_Projection_Prev/25Steps/" + Domains[2] + args.loss_str + "_25.pt", map_location=map_location)
dt_files3 = torch.load("Domain_Projection_Prev/25Steps/" + Domains[3] + args.loss_str + "_25.pt", map_location=map_location)
n_min = np.min([len(dt_files1), len(dt_files2), len(dt_files3)])
ds_list = list(ds_files.keys())
dt_list1 = list(dt_files1.keys())
dt_list2 = list(dt_files2.keys())
dt_list3 = list(dt_files3.keys())
dst_latent, ds_latent, dt_latent1, dt_latent2,  dt_latent3 = [], [], [], [], []
for i in range(n_min):
    key = ds_list[i]
    dst_latent.append(ds_files[key]['latent'].view(-1))
    ds_latent.append(ds_files[key]['latent'].view(-1))
    # dst_latent.append(torch.mean(ds_files[key]['latent'], dim=0))
    # ds_latent.append(torch.mean(ds_files[key]['latent'], dim=0))

    key = dt_list1[i]
    dst_latent.append(dt_files1[key]['latent'].view(-1))
    dt_latent1.append(dt_files1[key]['latent'].view(-1))
    key = dt_list2[i]
    dst_latent.append(dt_files2[key]['latent'].view(-1))
    dt_latent2.append(dt_files2[key]['latent'].view(-1))
    key = dt_list3[i]
    dst_latent.append(dt_files3[key]['latent'].view(-1))
    dt_latent3.append(dt_files3[key]['latent'].view(-1))
    # dst_latent.append(torch.mean(dt_files[key]['latent'], dim=0))
    # dt_latent.append(torch.mean(dt_files[key]['latent'], dim=0))

ds_latent = torch.stack(ds_latent)
# ds_mean = torch.mean(ds_latent, dim=0)
dt_latent1 = torch.stack(dt_latent1)
dt_latent2 = torch.stack(dt_latent2)
dt_latent3 = torch.stack(dt_latent3)

# dt_mean = torch.mean(dt_latent, dim=0)
# mean_dist = np.linalg.norm(ds_mean-dt_mean, ord=2)

dst_latent = torch.stack(dst_latent)
pca = PCA(n_components=2)
pca.fit(dst_latent)
ds_points = pca.transform(ds_latent)
dt_points1 = pca.transform(dt_latent1)
dt_points2 = pca.transform(dt_latent2)
dt_points3 = pca.transform(dt_latent3)

plt.subplots()
plt.scatter(ds_points[:, 0], ds_points[:, 1], c='r', label=args.ds)
plt.scatter(dt_points1[:, 0], dt_points1[:, 1], c='b', label=Domains[1])
plt.scatter(dt_points2[:, 0], dt_points2[:, 1], c='g', label=Domains[2])
plt.scatter(dt_points3[:, 0], dt_points3[:, 1], c='y', label=Domains[3])
plt.legend()
plt.title(f'{args.loss_str}')
plt.savefig(f'{args.loss_str}_merge.png')
plt.show()

pca = PCA(n_components=16)
pca.fit(dst_latent)
exp_var_pca = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)  # cumulative sum of variance explained with [n] features
# Plot the explained variance against cumulative explained variance
plt.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
         label='Cumulative explained variance')
plt.ylabel("Explained variance ratio")
plt.xlabel(f"PC index, loss={args.loss_str}")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(f"Variance_ratio, loss={args.loss_str}_merge.png")
plt.show()
