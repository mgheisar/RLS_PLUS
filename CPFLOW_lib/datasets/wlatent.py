"""
Copyright (c) 2017, George Papamakarios
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of anybody else.
"""
import os

import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class WLATENT:
    class Data:

        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, root, load_stats=False):

        file = os.path.join(root, "w_samples_train_face1024.pt")
        # file = os.path.join(root, "w_samples_train_face.pt")
        if load_stats:
            data_train, data_validate, data_test = load_data(file)
            data = np.vstack((data_train, data_validate))
            self.mu = data.mean(axis=0)
            self.std = data.std(axis=0)
            self.n_dims = data.shape[1]
        else:
            trn, val, tst = load_data_normalised(file)

            self.trn = self.Data(trn)
            self.val = self.Data(val)
            self.tst = self.Data(tst)

            trn_n, val_n, tst_n = load_data(file)
            self.trn_n = self.Data(trn_n)
            self.val_n = self.Data(val_n)
            self.tst_n = self.Data(tst_n)
            self.n_dims = self.trn.x.shape[1]


def load_data(root_path):
    # NOTE: To remember how the pre-processing was done.
    # data = pd.read_csv(root_path, names=[str(x) for x in range(50)], delim_whitespace=True)
    # print data.head()
    # data = data.as_matrix()
    # # Remove some random outliers
    # indices = (data[:, 0] < -100)
    # data = data[~indices]
    #
    # i = 0
    # # Remove any features that have too many re-occuring real values.
    # features_to_remove = []
    # for feature in data.T:
    #     c = Counter(feature)
    #     max_count = np.array([v for k, v in sorted(c.iteritems())])[0]
    #     if max_count > 5:
    #         features_to_remove.append(i)
    #     i += 1
    # data = data[:, np.array([i for i in range(data.shape[1]) if i not in features_to_remove])]
    # np.save("~/data/miniboone/data.npy", data)
    map_location = lambda storage, loc: storage.cpu()
    data = torch.load(root_path, map_location=map_location)
    # data = torch.nn.LeakyReLU(5)(data)
    data = data.cpu().numpy()
    # data = data[:100000, :128]
    N_test = int(0.1 * data.shape[0])

    data_validate = data[-N_test:, :]
    data_test = data_validate
    # data_test = data[-N_test:, :]
    # data = data[0:-N_test, :]
    # N_validate = int(0.1 * data.shape[0])
    # data_validate = data[-N_validate:, :]
    # data_train = data[0:-N_validate, :]

    data_train = data[0:-N_test, :]
    return data_train, data_validate, data_test


def load_data_normalised(root_path):
    data_train, data_validate, data_test = load_data(root_path)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test


if __name__ == "__main__":
    import CPFLOW_lib.datasets as datasets
    import pickle

    wlatent = WLATENT(datasets.root, load_stats=True)
    wlatent = {"mu": wlatent.mu, "std": wlatent.std, "n_dims": wlatent.n_dims}
    with open('wlatent_face1024.pkl', 'wb') as f:
        pickle.dump(wlatent, f)
