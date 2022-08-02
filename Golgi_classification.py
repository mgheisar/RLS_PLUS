import glob
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skimage.filters import gaussian
from findmaxima2d import find_maxima, find_local_maxima
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import random
import time


# parser = argparse.ArgumentParser(description="Projection with PULSE")
# parser.add_argument("--n_tol", type=int, default=10, help="local maxima parameter")
# args = parser.parse_args()


def process(n_tol, sigma):
    np.random.seed(0)
    data_x, data_y = [], []
    n_c = 2000
    input_dir_HR = 'input/superres/HR/'
    size = 128
    # input_dir = '/projects/imagesets/Golgi/golgi/196x196/'  # 'input/superres/HR/'
    for label in range(2):
        files = sorted(glob.glob(os.path.join(input_dir_HR, str(label), '*.png')))
        # index = random.sample(range(len(files)), n_c)
        for i in range(n_c):
            imag = Image.open(files[i])
            # imag = Image.open(files[index[i]])
            # width, height = imag.size  # Get dimensions
            # left = (width - size) / 2
            # top = (height - size) / 2
            # right = (width + size) / 2
            # bottom = (height + size) / 2
            # imag = imag.crop((left, top, right, bottom))

            img1 = np.array(imag)
            # img = img1
            img = gaussian(img1, sigma=sigma, multichannel=True, preserve_range=True)
            img_data = (np.sum(img, 2) / 3.0)
            # Finds the local maxima using maximum filter.
            local_max = find_local_maxima(img_data)
            # Finds the maxima.
            y_max, x_max, regs = find_maxima(img_data, local_max, ntol=n_tol)
            data_x.append([len(x_max)])
            data_y.append(label)

    X_train = np.stack(data_x, 0)
    Y_train = np.stack(data_y, 0)
    # X_train, X_test, Y_train, Y_test = train_test_split(data_x, data_y, test_size=0.2)
    classifier = SVC(kernel='rbf')  # 'linear'
    classifier.fit(X_train, Y_train)
    data_x_bic, data_y_bic, data_x_lr, data_y_lr = [], [], [], []
    data_x_sr, data_y_sr, img_name = [], [], []
    n_c = 1100
    input_dir_LR = 'input/superres/LR/'
    input_dir_SR = 'input/superres/SR/'
    for label in range(2):
        files_SR = sorted(glob.glob(os.path.join(input_dir_SR, str(label), '*.png')))
        for i in range(n_c):
            imag = Image.open(files_SR[i])
            imag = imag.resize((size, size), resample=Image.BICUBIC)
            img1 = np.array(imag)
            # img = img1
            img = gaussian(img1, sigma=sigma, multichannel=True, preserve_range=True)
            img_data = (np.sum(img, 2) / 3.0)
            # Finds the local maxima using maximum filter.
            local_max = find_local_maxima(img_data)
            # Finds the maxima.
            y_max, x_max, regs = find_maxima(img_data, local_max, ntol=n_tol)
            data_x_sr.append([len(x_max)])
            data_y_sr.append(label)

            file_LR = files_SR[i].split('_10*L1')[0].split('/')[-1]
            img_name.append(file_LR)
            file_LR = input_dir_LR + str(label) + '/' + file_LR + '.png'
            imag = Image.open(file_LR)
            img1 = np.array(imag)
            # img = img1
            img = gaussian(img1, sigma=sigma, multichannel=True, preserve_range=True)
            img_data = (np.sum(img, 2) / 3.0)
            # Finds the local maxima using maximum filter.
            local_max = find_local_maxima(img_data)
            # Finds the maxima.
            y_max, x_max, regs = find_maxima(img_data, local_max, ntol=n_tol)
            data_x_lr.append([len(x_max)])
            data_y_lr.append(label)

            imag = Image.open(file_LR)
            imag = imag.resize((size, size), resample=Image.BICUBIC)
            img1 = np.array(imag)
            # img = img1
            img = gaussian(img1, sigma=sigma, multichannel=True, preserve_range=True)
            img_data = (np.sum(img, 2) / 3.0)
            # Finds the local maxima using maximum filter.
            local_max = find_local_maxima(img_data)
            # Finds the maxima.
            y_max, x_max, regs = find_maxima(img_data, local_max, ntol=n_tol)
            data_x_bic.append([len(x_max)])
            data_y_bic.append(label)

    X_test_lr = np.stack(data_x_lr, 0)
    Y_test_lr = np.stack(data_y_lr, 0)
    Y_pred_lr = classifier.predict(X_test_lr)

    X_test_bic = np.stack(data_x_bic, 0)
    Y_test_bic = np.stack(data_y_bic, 0)
    Y_pred_bic = classifier.predict(X_test_bic)

    X_test_sr = np.stack(data_x_sr, 0)
    Y_test_sr = np.stack(data_y_sr, 0)
    Y_pred_sr = classifier.predict(X_test_sr)
    inds = np.where(Y_test_sr == Y_pred_sr)[0]
    print(len(inds))
    # for j in range(len(inds)):
    #     i = inds[j]
    #     if not (Y_test_bic[i] == Y_pred_bic[i]):
    #         image_LR = Image.open(input_dir_LR + str(data_y_lr[i]) + '/' + img_name[i] + '.png')
    #         image_SR = Image.open(input_dir_SR + str(data_y_lr[i]) + '/' + img_name[i] + '_10*L1+1*Adv.png')
    #         image_HR = Image.open(input_dir_HR + str(data_y_lr[i]) + '/' + img_name[i].split('-')[-1] + '.png')
    #         image_bic = image_LR.resize((size, size), resample=Image.BICUBIC)
    #         dict = {"LR": image_LR, "SR": image_SR, "HR": image_HR, "bic": image_bic}
    #         for key, image in dict.items():
    #             img1 = np.array(image)
    #             img = gaussian(img1, sigma=2, multichannel=True, preserve_range=True)
    #             img_data = (np.sum(img, 2) / 3.0)
    #             # Finds the local maxima using maximum filter.
    #             local_max = find_local_maxima(img_data)
    #             # Finds the maxima.
    #             y, x, regs = find_maxima(img_data, local_max, ntol=30)
    #             print(f'{img_name[i]} {key}: {len(x)}')
    #             fig = plt.figure()
    #             plt.axis('off')
    #             plt.imshow(img1)
    #             plt.scatter(x, y, s=100, marker='x', color='r', linewidth=6)
    #             plt.savefig(f"input/superres/maxima/{img_name[i]}_{key}.png")
    #             plt.close()
    return n_tol, accuracy_score(Y_test_lr, Y_pred_lr), accuracy_score(Y_test_bic, Y_pred_bic), accuracy_score(
        Y_test_sr, Y_pred_sr)
    # print(confusion_matrix(Y_test, Y_pred))


results = process(n_tol=16, sigma=0.1)
print(f'results {results}')
