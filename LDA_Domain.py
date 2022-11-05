import argparse
import torch
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from mpl_toolkits import mplot3d

np.random.seed(0)
# --exp "3" --ds "DMSO" --dt "cytochalasin D_10.0"
# cytochalasin B_10.0  demecolcine_10.0 staurosporine_0.01 cytochalasin D_10.0
# demecolcine_10.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description="Computing PCA on latent vectors of DS and DT")
    parser.add_argument("--ds", type=str, help="name of the source domain")
    parser.add_argument("--dt", type=str, help="name of the target domain")
    parser.add_argument('--exp', type=str, default='3', help='experiment name')
    args = parser.parse_args()
    # Domains = ['DMSO_656', 'cytochalasin B_10.0', 'cytochalasin D_3.0', 'demecolcine_10.0']
    # Domains = ['DMSO_656', 'cytochalasin B_0.01', 'cytochalasin D_0.003', 'demecolcine_0.003']
    Domains = ['DMSO_656', 'demecolcine_0.003', 'demecolcine_0.01', 'demecolcine_0.03', 'demecolcine_10.0',
               'cytochalasin B_0.01', 'cytochalasin B_0.1', 'cytochalasin B_0.3', 'cytochalasin B_1.0',
               'cytochalasin B_3.0', 'cytochalasin B_10.0',
               'cytochalasin D_0.003', 'cytochalasin D_0.01', 'cytochalasin D_0.3', 'cytochalasin D_3.0',
               'staurosporine_0.003']
    args.ds = Domains[0]
    Loss_vec = ["1*L1", "1*L1+0.1*Percept", "1*L1+0.1*Percept+0.1*Adv", "1*L1+0.01*Adv+0.1*Percept"]

    latent_dir = "Domain_Projection_Prev/500Steps_/"
    visualization = True
    args.loss_str = Loss_vec[2]

    ds_files = torch.load(latent_dir + args.ds + args.loss_str + ".pt", map_location=device)
    dt_files = []
    for i in range(len(Domains) - 1):
        dt_files.append(torch.load(latent_dir + Domains[i + 1] + args.loss_str + ".pt", map_location=device))
    n_min = np.min([len(dt) for dt in dt_files])

    ds_list = list(ds_files.keys())
    dt_list = [list(dt.keys()) for dt in dt_files]
    dst_latent, ds_latent = [], []
    dt_latent = [[] for i in range(len(dt_list))]
    labels = []
    for i in range(n_min):
        key = ds_list[i]
        dst_latent.append(ds_files[key]['latent'].view(-1))
        labels.append(0)
        ds_latent.append(ds_files[key]['latent'].view(-1))
        # dst_latent.append(torch.mean(ds_files[key]['latent'], dim=0))
        # ds_latent.append(torch.mean(ds_files[key]['latent'], dim=0))
        for j in range(len(dt_list)):
            key = dt_list[j][i]
            dst_latent.append(dt_files[j][key]['latent'].view(-1))
            labels.append(j + 1)
            dt_latent[j].append(dt_files[j][key]['latent'].view(-1))

    labels = np.stack(labels, 0)
    ds_latent = torch.stack(ds_latent)
    dt_latent = [torch.stack(dt) for dt in dt_latent]
    dst_latent = torch.stack(dst_latent)
    #  # ----------------- Classification ----------------------------------
    model = LinearDiscriminantAnalysis()
    if visualization:
        X_train = dst_latent
        Y_train = labels
        X_test, Y_test = X_train, Y_train
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(dst_latent, labels, test_size=0.2)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

    model.fit(X_train, Y_train)
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)

    classifier = SVC(kernel='rbf')
    classifier.fit(X_train, Y_train)
    if not visualization:
        Y_pred = classifier.predict(X_test)
        print(f'SVM Test accuracy: {accuracy_score(Y_test, Y_pred)}')
        print(f'confusion matrix: {confusion_matrix(Y_test, Y_pred)}')
    Y_pred = classifier.predict(X_train)
    print(f'SVM Train accuracy: {accuracy_score(Y_train, Y_pred)}')
    print(f'confusion matrix: {confusion_matrix(Y_train, Y_pred)}')

    # #  # ----------------- Visualization----------------------------------
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
    if visualization:
        plt.figure()
        ax = plt.axes(projection='3d')
        plt.subplots_adjust(bottom=0.42)
        for i in range(len(Domains)):
            ax.scatter3D(X_train[Y_train == i, 0], X_train[Y_train == i, 1], X_train[Y_train == i, 2], label=Domains[i])
        plt.legend(loc='upper left', bbox_to_anchor=(-0.2, -0.05),
                   fancybox=True, shadow=True, ncol=2)
        # plt.title(f'{args.loss_str}')
        plt.savefig(f'{args.loss_str}_all_.png')
        plt.show()

