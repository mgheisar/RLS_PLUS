import torch
from torch.nn import functional as F
from bicubic import BicubicDownSample
import lpips
import numpy as np


class LossBuilder(torch.nn.Module):
    def __init__(self, ref_im, ref_pred, loss_str, factor=16, device='cuda:0',
                 normalize=False, gpu_num=0):
        super(LossBuilder, self).__init__()
        assert ref_im.shape[2] == ref_im.shape[3]
        self.ref_im = ref_im
        self.ref_pred = ref_pred
        self.device = device
        self.D = BicubicDownSample(factor=factor, device=device, normalize=normalize)
        self.parsed_loss = [loss_term.split('*') for loss_term in loss_str.split('+')]
        self.perceptual = lpips.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=device.startswith("cuda"), gpu_ids=[int(gpu_num)]
        )
        data = np.load('inversion_stats.npz')
        self.gt_mean = torch.tensor(data['mean']).to(device).view(1, -1).float()
        self.gt_conv = torch.tensor(data['cov']).to(device)
    # Takes a list of tensors, flattens them, and concatenates them into a vector
    # Used to calculate euclidian distance between lists of tensors
    def flatcat(self, l):
        l = l if(isinstance(l, list)) else [l]
        return torch.cat([x.flatten() for x in l], dim=0)

    def _loss_l2(self, gen_im, **kwargs):
        return F.mse_loss(gen_im, self.ref_im)

    def _loss_l1(self, gen_im, **kwargs):
        return F.l1_loss(gen_im, self.ref_im)

    def _percept(self, gen_im, **kwargs):
        return self.perceptual(gen_im, self.ref_im).sum()

    def _gauusian_loss(self, latent):
        loss = (latent - self.gt_mean) @ self.gt_cov_inv @ (latent - self.gt_mean).transpose(1, 0)
        return loss.mean()

    def _noise_regularize(self, noises, **kwargs):
        loss = 0
        for noise in noises:
            size = noise.shape[2]

            while True:
                loss = (
                        loss
                        + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                        + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
                )

                if size <= 8:
                    break

                noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
                noise = noise.mean([3, 5])
                size //= 2
        return loss

    # Uses geodesic distance on sphere to sum pairwise distances of the 18 vectors
    def _loss_geocross(self, latent, **kwargs):
        if len(latent.shape) == 2:
            return 0
        else:
            X = latent.view(-1, 1, latent.shape[1], latent.shape[2])
            Y = latent.view(-1, latent.shape[1], 1, latent.shape[2])
            A = ((X-Y).pow(2).sum(-1)+1e-9).sqrt()
            B = ((X+Y).pow(2).sum(-1)+1e-9).sqrt()
            D = 2*torch.atan2(A, B)
            D = ((D.pow(2)*latent.shape[2]).mean((1, 2))/8.).sum()
            return D

    def _loss_adv(self, fake_pred, **kwargs):
        if self.ref_pred is None:
            loss = F.softplus(-fake_pred).mean()
        else:
            loss = torch.abs(F.softplus(-fake_pred)-F.softplus(-self.ref_pred)).mean()
        return loss

    def forward(self, latent, gen_im, fake_pred, noises):
        # a, b = torch.min(self.D(gen_im)), torch.max(self.D(gen_im))
        # c, d = torch.min(self.ref_im), torch.max(self.ref_im)
        if self.ref_pred is None:
            var_dict = {'latent': latent,
                        'gen_im': self.D(gen_im),  # self.D(gen_im)
                        'fake_pred': fake_pred,
                        'noises': noises,
                        }
        else:
            var_dict = {'latent': latent,
                        'gen_im': gen_im,  # self.D(gen_im)
                        'fake_pred': fake_pred,
                        'noises': noises,
                        }
        loss = 0
        loss_fun_dict = {
            'L2': self._loss_l2,
            'L1': self._loss_l1,
            'GEOCROSS': self._loss_geocross,
            'Percept': self._percept,
            'Adv': self._loss_adv,
            'Noise_loss': self._noise_regularize,
        }
        losses = {}
        for weight, loss_type in self.parsed_loss:
            tmp_loss = loss_fun_dict[loss_type](**var_dict)
            losses[loss_type] = tmp_loss
            loss += float(weight)*tmp_loss
        return loss, losses
