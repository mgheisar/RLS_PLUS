# # implementation of PULSE with NF Gaussianization
import os
from pathlib import Path
import torchvision
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from augs import (AddBlur, AddMotionBlur, AddGaussianNoise, AddOcclusion, ChangeBrightness,
                  ChangeContrast, RandomPerspective, GaussianBlur, SaltAndPepperNoise)
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, \
    ColorJitter, RandomGrayscale, ToTensor, Compose
from bicubic import BicubicDownSample
from kornia.filters import MotionBlur


toPIL = torchvision.transforms.ToPILImage()
out_dir = 'input/project/resSR'

def get_transforms(augs):
    # input_size = args.input_size
    # if args.phase == 'test':
    #
    #     if args.pretrained:
    #         transforms = Compose([Resize(input_size), CenterCrop(input_size), ToTensor(),
    #                               Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #     else:
    #         transforms = Compose([Resize(input_size), CenterCrop(input_size), ToTensor()])
    # elif args.phase == 'train':
    transform_list = []
    if augs:
        if 'rotate' in augs:
            transform_list.append(RandomRotation(100, expand=False))
        if 'hflip' in augs:
            transform_list.append(RandomHorizontalFlip(p=1))
        if 'vflip' in augs:
            transform_list.append(RandomVerticalFlip(p=1))
        if 'contrast' in augs:
            transform_list.append(ChangeContrast())
        if 'brightness' in augs:
            transform_list.append(ChangeBrightness())
        if 'occlusion' in augs:
            transform_list.append(AddOcclusion())
        if 'regularblur' in augs:
            transform_list.append(AddBlur())
        # if 'motionblur' in augs:
        #     transform_list.append(AddMotionBlur(length=100))  # 9
        if 'gaussianblur' in augs:
            transform_list.append(torchvision.transforms.RandomApply([GaussianBlur(1)], p=1.0))
        if 'defocusblur' in augs:
            transform_list.append(AddDefocusBlur())
        if 'perspective' in augs:
            transform_list.append(RandomPerspective(p=1))
        if 'colorjitter' in augs:
            transform_list.append(ColorJitter(brightness=0, contrast=0, saturation=2, hue=0.05))
        if 'gray' in augs:
            transform_list.append(RandomGrayscale(p=1))
    # transform_list.append(Resize(input_size))
    transform_list.append(ToTensor())

    if augs and 'gaussiannoise' in augs:
        transform_list.append(AddGaussianNoise(mean=0, sigma=0.1))
    if augs and 'saltpepper' in augs:
        transform_list.append(SaltAndPepperNoise(d=0.05))
    # if args.pretrained:
    #     transform_list.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    transforms = Compose(transform_list)
    return transforms


class Images(Dataset):
    def __init__(self, image_list, factor=64, duplicates=1, aug=None):
        # args.files = [sorted(glob.glob(f"input/project/inputt/*_{args.factor}x.jpg"))[args.img_idx]]
        self.image_list = image_list
        self.duplicates = duplicates  # duplicate the image in the dataset to produce multiple HR images
        self.aug = aug
        self.factor = factor
        self.transform = get_transforms(self.aug)

        self.Downsampler = BicubicDownSample(factor=self.factor)

    def __len__(self):
        return self.duplicates * len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx // self.duplicates]
        ToPIL = torchvision.transforms.ToPILImage()
        im_name = Path(img_path).stem.split('_')[0]
        # out_dir = 'input/project/resSR/robustness/lr'
        if self.aug is None:
            transform = torchvision.transforms.Compose([torchvision.transforms.Resize(int(1024/self.factor),
                                                      interpolation=Image.BICUBIC), torchvision.transforms.ToTensor()])
            image = transform(Image.open(img_path)).to(torch.device("cuda"))
            # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            # image = transform(Image.open(img_path)).to(torch.device("cuda"))
            # image = self.Downsampler(image.unsqueeze(0))[0]
        elif 'motionblur' in self.aug:
            # image = self.transform(Image.open(os.path.join("input/project/resHR", im_name + "_64x_HR.jpg"))).to(
            #     torch.device('cuda'))
            image = Image.open(os.path.join("input/project/resHR", im_name + "_64x_HR.jpg"))
            image = self.transform(image).to(torch.device('cuda'))
            motion_blur = MotionBlur(kernel_size=49, angle=45, direction=1)
            image = motion_blur(image.unsqueeze(0)).squeeze(0)
            # img = toPIL(image.cpu().detach().clamp(0, 1))
            # img.save(f"{out_dir}/{im_name}_{elf.aug[0]}_hr{self.factor}.jpg")
            image = self.Downsampler(image.unsqueeze(0))[0]
            # img = toPIL(image.cpu().detach().clamp(0, 1))
            # img = img.resize((1024, 1024), Image.NEAREST)
            # img.save(f"{out_dir}/{im_name}_{self.aug[0]}_lr{self.factor}.jpg")
        elif self.aug:
            image = self.transform(Image.open(img_path)).to(torch.device("cuda"))
            # img = toPIL(image.cpu().detach().clamp(0, 1))
            # img = img.resize((1024, 1024), Image.NEAREST)
            # img.save(f"{out_dir}/{im_name}_{self.aug[0]}_lr{self.factor}.jpg")
        image_hr = []
        # image_hr = torchvision.transforms.ToTensor()(Image.open(img_path.split('_')[0] + '_HR.jpg')).to(
        # torch.device("cuda"))
        if self.duplicates == 1:
            return image, image_hr, os.path.splitext(os.path.basename(img_path))[0]
        else:
            return image, image_hr, os.path.splitext(os.path.basename(img_path))[0] + f"_{(idx % self.duplicates) + 1}"
