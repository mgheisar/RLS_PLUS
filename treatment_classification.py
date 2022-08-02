import argparse
import torch
from model import Generator, Discriminator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
import os
import glob
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision import utils

torch.cuda.set_device(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.deterministic = True
np.random.seed(1)


def make_image(tensor):
    return (
        tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .permute(0, 2, 3, 1)
            .to("cpu")
            .numpy()
    )


if __name__ == "__main__":
    torch.set_grad_enabled(False)  # cytochalasin B_0.01 cytochalasin B_10.0 staurosporine_0.003 demecolcine_10.0
    parser = argparse.ArgumentParser(description="Domain Translation")
    parser.add_argument("--ckpt", type=str, default="checkpoint/BBBC021/150000.pt",
                        help="stylegan2 checkpoints")
    parser.add_argument("--size", type=int, default=128, help="output image size of the generator")
    parser.add_argument("-n", "--n_sample", type=int, default=30, help="number of samples created")
    parser.add_argument("--ds", type=str, default="DMSO_656", help="name of the source domain")
    parser.add_argument("--dt", type=str, default="demecolcine_10.0", help="name of the target domain")

    args = parser.parse_args()
    map_location = lambda storage, loc: storage.cpu()
    Loss_vec = ["1*L1", "1*L1+0.1*Percept", "1*L1+0.1*Percept+0.1*Adv", "1*L1+1*Percept+0.1*Adv"]
    # args.loss_str = Loss_vec[2]
    # latent_dir = "Domain_Projection_Prev/500Steps_/"

    args.loss_str = Loss_vec[3]
    latent_dir = "Domain_Projection_Prev/1000Steps_without_noise/"

    ds_files = torch.load(latent_dir + args.ds + args.loss_str + ".pt", map_location=map_location)
    dt_files = torch.load(latent_dir + args.dt + args.loss_str + ".pt", map_location=map_location)

    g = Generator(args.size, 512, 8).to(device)
    map_location = lambda storage, loc: storage.cuda()
    g.load_state_dict(torch.load(args.ckpt, map_location=map_location)["g_ema"], strict=False)
    g.eval()
    discriminator = Discriminator(args.size).to(device)
    discriminator.load_state_dict(torch.load(args.ckpt, map_location=map_location)["d"], strict=False)
    noises_single = g.make_noise()
    ds_noises, dt_noises, noises = [], [], []
    args.n_sample = np.min([len(ds_files), len(dt_files)])
    batch_size = 64
    for noise in noises_single:
        ds_noises.append(noise.repeat(args.n_sample, 1, 1, 1).normal_())
        dt_noises.append(noise.repeat(args.n_sample, 1, 1, 1).normal_())
        noises.append(noise.repeat(batch_size, 1, 1, 1).normal_())

    ds_list = list(ds_files.keys())
    dt_list = list(dt_files.keys())

    ds_latent, dt_latent = [], []
    for i in range(args.n_sample):
        key = ds_list[i]
        ds_latent.append(ds_files[key]['latent'])
        n_i = 0
        for noise in ds_files[key]['noise']:
            ds_noises[n_i][i, :] = noise[0]
            n_i += 1

        key = dt_list[i]
        dt_latent.append(dt_files[key]['latent'])
        n_i = 0
        for noise in dt_files[key]['noise']:
            dt_noises[n_i][i, :] = noise[0]
            n_i += 1

    ds_latent = torch.stack(ds_latent).to(device)
    dt_latent = torch.stack(dt_latent).to(device)

    ds_mean = torch.median(ds_latent, dim=0)[0].to(device)
    dt_mean = torch.median(dt_latent, dim=0)[0].to(device)

    # ds_mean = torch.mean(ds_latent, dim=0)[0].to(device)
    # dt_mean = torch.mean(dt_latent, dim=0)[0].to(device)

    ds_dt = ds_mean - dt_mean
    dt_ds = dt_mean - ds_mean

    nn = 0
    degree = 1
    if nn > 0:
        dist = torch.norm(dt_latent - dt_mean, dim=(1, 2), p=None)
        knn_t = dist.topk(nn, largest=False).indices[-1]
        dist = torch.norm(ds_latent - ds_mean, dim=(1, 2), p=None)
        knn_s = dist.topk(nn, largest=False).indices[-1]
        # knn_t = np.random.randint(args.n_sample)
        # knn_s = knn_t
        dt_ds = dt_latent[knn_t] - ds_latent[knn_s]
    # print('norm:', torch.mean(torch.norm(dt_ds, p=2.0, dim=1)))
    # dt_ds = torch.nn.functional.normalize(dt_ds, p=2.0, dim=1)

    img_s, img_t, img_st = [], [], []
    fake_s, fake_st = [], []
    for i in range(args.n_sample // batch_size):
        l_s = ds_latent[i * batch_size:(i + 1) * batch_size]
        l_t = dt_latent[i * batch_size:(i + 1) * batch_size]

        # for n_i in range(len(noises)):
        #     noises[n_i] = ds_noises[n_i][i * batch_size:(i + 1) * batch_size]

        img, _ = g(
            [l_s],
            input_is_latent=True,
            noise=noises
        )
        fake_s.append(torch.mean(discriminator(img)))
        img_s.append(make_image(img).reshape(batch_size, -1))

        # for n_i in range(len(noises)):
        #     noises[n_i] = dt_noises[n_i][i * batch_size:(i + 1) * batch_size]

        img, _ = g(
            [l_t],
            input_is_latent=True,
            noise=noises
        )
        img_t.append(make_image(img).reshape(batch_size, -1))

        img, _ = g(
            [l_s + degree * dt_ds],
            input_is_latent=True,
            noise=noises
        )
        fake_st.append(torch.mean(discriminator(img)))
        img_st.append(make_image(img).reshape(batch_size, -1))

    print('fake s', torch.mean(torch.stack(fake_s)))
    print('fake st', torch.mean(torch.stack(fake_st)))
    input_dir = "/projects/superres/Marzieh/BBBC021"
    files_s = sorted(glob.glob(os.path.join(input_dir, args.ds, '*.png')))
    files_t = sorted(glob.glob(os.path.join(input_dir, args.dt, '*.png')))
    X_train, Y_train = [], []
    nc = np.min([len(files_s), len(files_t)])
    size = 128
    for i in range(nc):
        img = Image.open(files_s[i])
        # img_data = (np.sum(img, 2) / 3.0)
        width, height = img.size  # Get dimensions
        left = (width - size) / 2
        top = (height - size) / 2
        right = (width + size) / 2
        bottom = (height + size) / 2
        img = np.array(img.crop((left, top, right, bottom)))
        X_train.append(img.flatten())
        Y_train.append(0)

        img = Image.open(files_t[i])
        width, height = img.size  # Get dimensions
        left = (width - size) / 2
        top = (height - size) / 2
        right = (width + size) / 2
        bottom = (height + size) / 2
        img = np.array(img.crop((left, top, right, bottom)))
        # img_data = (np.sum(img, 2) / 3.0)
        X_train.append(img.flatten())
        Y_train.append(1)
    X_train = np.stack(X_train, 0)
    Y_train = np.stack(Y_train, 0)

    img_s, img_t, img_st = np.stack(img_s), np.stack(img_t), np.stack(img_st)

    img_s = img_s.reshape(-1, img_s.shape[2])
    img_t = img_t.reshape(-1, img_t.shape[2])
    img_st = img_st.reshape(-1, img_st.shape[2])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    print('PCA')
    pca = PCA(n_components=512)
    X_train_pca = pca.fit_transform(X_train)

    classifier = SVC(kernel='rbf')
    classifier.fit(X_train_pca, Y_train)
    Y_pred = classifier.predict(X_train_pca)
    print(f'SVM Train accuracy DMSO vs compound: {accuracy_score(Y_train, Y_pred)}')
    # print(f'confusion matrix: {confusion_matrix(Y_train, Y_pred)}')

    X_test = np.concatenate([img_s, img_t], 0)
    Y_test = np.concatenate([np.zeros((img_s.shape[0],)), np.ones((img_t.shape[0],))])
    X_test = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test)
    Y_pred = classifier.predict(X_test_pca)
    print(f'SVM Test accuracy Fake DMSO vs Fake Compound: {accuracy_score(Y_test, Y_pred)}')
    # print(f'confusion matrix: {confusion_matrix(Y_test, Y_pred)}')

    X_test = np.concatenate([img_s, img_st], 0)
    Y_test = np.concatenate([np.zeros((img_s.shape[0],)), np.ones((img_t.shape[0],))])
    X_test = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test)
    Y_pred = classifier.predict(X_test_pca)
    print(f'SVM Test accuracy DMSO vs its translation: {accuracy_score(Y_test, Y_pred)}')
    # print(f'confusion matrix: {confusion_matrix(Y_test, Y_pred)}')

    X_test = np.concatenate([img_st, img_t], 0)
    X_test = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test)
    Y_pred = classifier.predict(X_test_pca)
    print(f'SVM Test accuracy DMSO translation vs compound: {accuracy_score(Y_test, Y_pred)}')
    # print(f'confusion matrix: {confusion_matrix(Y_test, Y_pred)}')

    img_s, img_t, img_st = [], [], []
    inx = np.random.randint(args.n_sample, size=batch_size)
    l_s = ds_latent[inx]
    l_t = dt_latent[inx]
    img_s, _ = g(
        [l_s],
        input_is_latent=True,
        noise=noises
    )
    img_t, _ = g(
        [l_t],
        input_is_latent=True,
        noise=noises
    )

    img_st, _ = g(
        [l_s + degree * dt_ds],
        input_is_latent=True,
        noise=noises
    )
    grid = utils.save_image(
        torch.cat([img_s, img_st], 0),
        f"{args.ds}_{args.dt}_st_med.png",
        normalize=True,
        range=(-1, 1),
        nrow=batch_size,
    )


