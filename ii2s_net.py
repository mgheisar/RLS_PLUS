import torch
from torch import nn
from model import Generator
import numpy as np
import os
from sklearn.decomposition import IncrementalPCA


class Net(nn.Module):

    def __init__(self, opts):
        super(Net, self).__init__()
        self.opts = opts
        self.generator = Generator(opts.size, 512, 8).cuda()
        # self.generator = Generator(opts.size, opts.latent, opts.n_mlp, channel_multiplier=opts.channel_multiplier)
        self.load_weights()
        self.load_PCA_model()

    def load_weights(self):
        map_location = lambda storage, loc: storage.cuda()
        gan = torch.load(self.opts.ckpt, map_location=map_location)
        self.generator.load_state_dict(gan["g_ema"], strict=False)

        for param in self.generator.parameters():
            param.requires_grad = False
        self.generator.eval()

    def build_PCA_model(self, PCA_path):

        with torch.no_grad():
            latent = torch.randn((1000000, 512), dtype=torch.float32)
            # latent = torch.randn((10000, 512), dtype=torch.float32)
            self.generator.style.cpu()
            pulse_space = torch.nn.LeakyReLU(5)(self.generator.style(latent)).numpy()
            self.generator.style.cuda()
        n_components = 512
        transformer = IncrementalPCA(n_components, whiten=False, batch_size=max(100, 5 * n_components))
        X_mean = pulse_space.mean(0)
        transformer.fit(pulse_space - X_mean)
        X_comp = transformer.components_
        X_stdev = np.sqrt(transformer.explained_variance_)  # already sorted
        X_var_ratio = transformer.explained_variance_ratio_
        np.savez(PCA_path, X_mean=X_mean, X_comp=X_comp, X_stdev=X_stdev, X_var_ratio=X_var_ratio)

    def load_PCA_model(self):
        PCA_path = self.opts.ckpt[:-3] + '_PCA.npz'

        if not os.path.isfile(PCA_path):
            self.build_PCA_model(PCA_path)

        PCA_model = np.load(PCA_path)
        self.X_mean = torch.from_numpy(PCA_model['X_mean']).float().cuda()
        self.X_comp = torch.from_numpy(PCA_model['X_comp']).float().cuda()
        self.X_stdev = torch.from_numpy(PCA_model['X_stdev']).float().cuda()

    # def make_noise(self):
    #     noises_single = self.generator.make_noise()
    #     noises = []
    #     for noise in noises_single:
    #         noises.append(noise.repeat(1, 1, 1, 1).normal_())
    #
    #     return noises

    def cal_p_norm_loss(self, latent_in):
        # latent_p_norm = (torch.nn.LeakyReLU(negative_slope=5)(latent_in) - self.X_mean).bmm(
        #     self.X_comp.T.unsqueeze(0)) / self.X_stdev
        latent_p_norm = torch.mm(torch.nn.LeakyReLU(negative_slope=5)(latent_in) - self.X_mean,
                                 self.X_comp.T) / self.X_stdev
        p_norm_loss = torch.mean(torch.norm(latent_p_norm, dim=1))
        # p_norm_loss = latent_p_norm.pow(2).mean()
        return p_norm_loss
