import torch
from torch import nn
from torch.nn import functional as F


class BicubicDownSample(nn.Module):
    def bicubic_kernel(self, x, a=-0.50):
        """
        This equation is exactly copied from the website below:
        https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic
        """
        abs_x = torch.abs(x)
        if abs_x <= 1.:
            return (a + 2.) * torch.pow(abs_x, 3.) - (a + 3.) * torch.pow(abs_x, 2.) + 1
        elif 1. < abs_x < 2.:
            return a * torch.pow(abs_x, 3) - 5. * a * torch.pow(abs_x, 2.) + 8. * a * abs_x - 4. * a
        else:
            return 0.0

    def __init__(self, factor=4, cuda=True, padding='reflect'):
        super().__init__()
        self.factor = factor
        size = factor * 4
        k = torch.tensor([self.bicubic_kernel((i - torch.floor(torch.tensor(size / 2)) + 0.5) / factor)
                          for i in range(size)], dtype=torch.float32)
        k = k / torch.sum(k)
        # k = torch.einsum('i,j->ij', (k, k))
        k1 = torch.reshape(k, shape=(1, 1, size, 1))
        self.k1 = torch.cat([k1, k1, k1], dim=0)
        k2 = torch.reshape(k, shape=(1, 1, 1, size))
        self.k2 = torch.cat([k2, k2, k2], dim=0)
        self.cuda = '.cuda' if cuda else ''
        self.padding = padding
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, nhwc=False, clip_round=False, byte_output=False):
        # x = torch.from_numpy(x).type('torch.FloatTensor')
        filter_height = self.factor * 4
        filter_width = self.factor * 4
        stride = self.factor

        pad_along_height = max(filter_height - stride, 0)
        pad_along_width = max(filter_width - stride, 0)
        filters1 = self.k1.type('torch{}.FloatTensor'.format(self.cuda))
        filters2 = self.k2.type('torch{}.FloatTensor'.format(self.cuda))

        # compute actual padding values for each side
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left

        # apply mirror padding
        if nhwc:
            x = torch.transpose(torch.transpose(
                x, 2, 3), 1, 2)   # NHWC to NCHW

        # downscaling performed by 1-d convolution
        x = F.pad(x, (0, 0, pad_top, pad_bottom), self.padding)
        x = F.conv2d(input=x, weight=filters1, stride=(stride, 1), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.)

        x = F.pad(x, (pad_left, pad_right, 0, 0), self.padding)
        x = F.conv2d(input=x, weight=filters2, stride=(1, stride), groups=3)
        if clip_round:
            x = torch.clamp(torch.round(x), 0.0, 255.)

        if nhwc:
            x = torch.transpose(torch.transpose(x, 1, 3), 1, 2)
        if byte_output:
            x.type('torch.ByteTensor'.format(self.cuda))
        else:
            return x


class BicubicUpsample(nn.Module):
    """ Bicubic upsampling function with similar behavior to that in TecoGAN-Tensorflow
        Note:
            This function is different from torch.nn.functional.interpolate and matlab's imresize
            in terms of the bicubic kernel and the sampling strategy.
        References:
            http://verona.fi-p.unam.mx/boris/practicas/CubConvInterp.pdf
            https://stackoverflow.com/questions/26823140/imresize-trying-to-understand-the-bicubic-interpolation
    """

    def __init__(self, scale_factor, a=-0.75):
        super(BicubicUpsample, self).__init__()

        # calculate weights (according to Eq.(6) in the reference paper)
        cubic = torch.FloatTensor([
            [0, a, -2 * a, a],
            [1, 0, -(a + 3), a + 2],
            [0, -a, (2 * a + 3), -(a + 2)],
            [0, 0, a, -a]
        ])

        kernels = [
            torch.matmul(cubic, torch.FloatTensor([1, s, s ** 2, s ** 3]))
            for s in [1.0 * d / scale_factor for d in range(scale_factor)]
        ]  # s = x - floor(x)

        # register parameters
        self.scale_factor = scale_factor
        self.register_buffer('kernels', torch.stack(kernels))  # size: (f, 4)

    def forward(self, input):
        n, c, h, w = input.size()
        f = self.scale_factor

        # merge n&c
        input = input.reshape(n * c, 1, h, w)

        # pad input (left, right, top, bottom)
        input = F.pad(input, (1, 2, 1, 2), mode='replicate')

        # calculate output (vertical expansion)
        kernel_h = self.kernels.view(f, 1, 4, 1).to(input.device)
        output = F.conv2d(input, kernel_h, stride=1, padding=0)
        output = output.permute(0, 2, 1, 3).reshape(n * c, 1, f * h, w + 3)

        # calculate output (horizontal expansion)
        kernel_w = self.kernels.view(f, 1, 1, 4).to(input.device)
        output = F.conv2d(output, kernel_w, stride=1, padding=0)
        output = output.permute(0, 2, 3, 1).reshape(n * c, 1, f * h, f * w)

        # split n&c
        output = output.reshape(n, c, f * h, f * w)

        return output