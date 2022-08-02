from pathlib import Path
import torch
from PIL import Image
import argparse
from torchvision import transforms
from torchvision.datasets import ImageFolder

# # ## Preparing LR images for PULSE
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Preparing LR images for PULSE"
#     )
#     parser.add_argument('--input_dir', type=str, default='input', help='input data directory')
#     parser.add_argument("--size", type=int, default=128, help="output HR image size")
#     parser.add_argument("--scale", type=int, default=2, help="Super resolution scale")
#     parser.add_argument("--exp", type=str, default="Golgi", help="face or cells")
#
#     args = parser.parse_args()
#     if args.exp == "Golgi":
#         args.size = 128
#         # load validations set images as HR
#         transform = transforms.Compose([transforms.CenterCrop(args.size), transforms.ToTensor(), ])
#         root_path = Path("/projects/imagesets/Golgi/golgi/196x196/")
#         dataset = ImageFolder(root_path, transform=transform)
#         golgi = torch.load("val_set_golgi.pt")
#         n_c = 1000
#         n_c0, n_c1 = 0, 0
#         i = -1
#         while i < len(golgi["val"]):
#             i += 1
#             index = golgi["val"][i]  # golgi["train"]
#             if index > len(dataset):
#                 continue
#             image = dataset.samples[index][0]
#             if dataset.samples[index][1] == 0:
#                 c = 0
#                 n_c0 += 1
#             elif dataset.samples[index][1] == 1:
#                 c = 1
#                 n_c1 += 1
#             if n_c0 > n_c and n_c1 > n_c:
#                 break
#             elif n_c0 > n_c and c == 0:
#                 continue
#             elif n_c1 > n_c and c == 1:
#                 continue
#             img = Image.open(image)
#             width, height = img.size  # Get dimensions
#             left = (width - args.size) / 2
#             top = (height - args.size) / 2
#             right = (width + args.size) / 2
#             bottom = (height + args.size) / 2
#             img = img.crop((left, top, right, bottom))
#             # img.save('input/superres/HR/%d/imgv%d.png' % (c, i))
#             d = int(args.size / args.scale)
#             img = img.resize((d, d), resample=Image.BICUBIC)
#             img.save('input/superres/LR_2x/%d/imgv%d.png' % (c, i))
#
#             img = img.resize((args.size, args.size), resample=Image.BICUBIC)
#             img.save('input/superres/SR_bic_2x/%d/imgv%d.png' % (c, i))

# from skimage import transform, io
# import numpy as np
# img_A = io.imread('input/superres/images/imgd.png')
# # #input_shape = np.array(img_A.shape[:-1])
# img_A = transform.resize(img_A, (4, 4), order=0)
# # # img_A = 255 * ((img_A - np.min(img_A)) / (np.max(img_A) - np.min(img_A)))  ##--yes: y : 1 or no: n
# io.imsave('input/superres/images/imgd_.png', img_A)

# image = Image.open('input/HR/img0.png').convert("RGB")
# image = transforms.ToTensor()(image)
# from bicubic import BicubicDownSample
# D = BicubicDownSample(factor=args.scale, device='cpu')
# image = torch.unsqueeze(image, 0)
# image_lr = D(image)
# image_lr = torch.squeeze(image_lr, 0)
# toPIL = transforms.ToPILImage()
# img = toPIL(image_lr.cpu().detach())
# img.save('input/D_bic_16x.png')
#
# img = Image.open('input/obama.jpg').convert("RGB")
# img = img.resize((256, 256), resample=Image.BILINEAR)
# img.save('input/obama_256.png')
img = Image.open('input/obama_256.png').convert("RGB")
img = img.resize((16, 16), resample=Image.BICUBIC)
img.save('input/obama_16.png')
# img = Image.open('input/superres/PIL_16x.png').convert("RGB")
# img = img.resize((128, 128), resample=Image.NEAREST)
# img.save('input/superres/nn_16x.png')
