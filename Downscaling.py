from pathlib import Path
import torch
from PIL import Image
import argparse
from torchvision import transforms
from torchvision.datasets import ImageFolder
from bicubic import BicubicDownSample

# ## Preparing LR images for PULSE
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preparing LR images for PULSE"
    )
    parser.add_argument('--input_dir', type=str, default='input/superres/HR_Golgi', help='input data directory')
    parser.add_argument('--output_dir', type=str, default='input/superres/LR_Golgi_32x', help='output data directory')
    parser.add_argument("--size", type=int, default=128, help="output HR image size")
    parser.add_argument("--scale", type=int, default=32, help="Super resolution scale")
    parser.add_argument("--exp", type=str, default="Golgi", help="face or cells")
    parser.add_argument("--prepare", type=bool, default=False, help="Prepare HR images")
    args = parser.parse_args()

    args = parser.parse_args()
    if args.exp == "Golgi" and args.prepare:
        args.size = 128
        # load validations set images as HR
        transform = transforms.Compose([transforms.CenterCrop(args.size), transforms.ToTensor(), ])
        root_path = Path("/projects/imagesets/Golgi/golgi/196x196/")
        dataset = ImageFolder(root_path, transform=transform)
        golgi = torch.load("val_set_golgi.pt")
        n_c = 1000
        n_c0, n_c1 = 0, 0
        i = -1
        while i < len(golgi["val"]):
            i += 1
            index = golgi["val"][i]  # golgi["train"]
            if index > len(dataset):
                continue
            image = dataset.samples[index][0]
            if dataset.samples[index][1] == 0:
                c = 0
                n_c0 += 1
            elif dataset.samples[index][1] == 1:
                c = 1
                n_c1 += 1
            if n_c0 > n_c and n_c1 > n_c:
                break
            elif n_c0 > n_c and c == 0:
                continue
            elif n_c1 > n_c and c == 1:
                continue
            img = Image.open(image)
            width, height = img.size  # Get dimensions
            left = (width - args.size) / 2
            top = (height - args.size) / 2
            right = (width + args.size) / 2
            bottom = (height + args.size) / 2
            img = img.crop((left, top, right, bottom))
            img.save('input/superres/HR_Golgi/%d/imgv%d.png' % (c, i))
            # d = int(args.size / args.scale)
            # img = img.resize((d, d), resample=Image.BICUBIC)
            # img.save('input/superres/LR_2x/%d/imgv%d.png' % (c, i))
            #
            # img = img.resize((args.size, args.size), resample=Image.BICUBIC)
            # img.save('input/superres/SR_bic_2x/%d/imgv%d.png' % (c, i))

    elif args.exp == "Golgi" and not args.prepare:
        for i_c in range(2):
            print(i_c)
            output_dir = Path(f"{args.output_dir}/{i_c}")
            output_dir.mkdir(parents=True, exist_ok=True)
            files = Path(f"{args.input_dir}/{i_c}").glob("*.png")
            for im in files:
                img = Image.open(im)
                width, height = img.size
                if args.scale:
                    D = BicubicDownSample(factor=args.scale)
                    img = transforms.ToTensor()(img).unsqueeze(0).cuda()
                    img_lr = D(img)[0].cpu().detach().clamp(0, 1)
                    img_lr = transforms.ToPILImage()(img_lr)
                    a = im.stem
                img_lr.save(Path(f"{args.output_dir}/{i_c}") / (im.stem + ".png"))
    else:
        print("Preparing LR images")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # files = Path(args.input_dir).glob("*.jpg")
        import glob
        image_list = sorted(glob.glob(f"{args.input_dir}/*.jpg"))[:2000]
        ii = 0
        for im in image_list:
            img = Image.open(im)
            a = Path(im).stem
            width, height = img.size
            if args.scale:
                D = BicubicDownSample(factor=args.scale)
                img = transforms.ToTensor()(img).unsqueeze(0).cuda()
                img_lr = D(img)[0].cpu().detach().clamp(0, 1)
                img_lr = transforms.ToPILImage()(img_lr)
            img_lr.save(Path(f"{output_dir}") / (Path(im).stem.split('_')[0] + f"_{args.scale}x.jpg"))


# scale = 64
# # im_vec = [22, 25, 76, 108, 136, 432, 723, 732, 734, 742, 743, 838, 842, 853, 20007]
# im_vec = [2,6,8,76,723,742,743,732]
# for ind in im_vec:
#     im = Path(f"input/project/resHR/{ind:05d}_64x_HR.jpg")
#     output_dir = f"input/project/lrr/lr_{scale}"
#     img = Image.open(im)
#     # img.save(Path(f"{output_dir}") / (im.stem.split('_')[0] + f"_HR.jpg"))
#     D = BicubicDownSample(factor=scale)
#     img = transforms.ToTensor()(img).unsqueeze(0).cuda()
#     img_lr = D(img)[0].cpu().detach().clamp(0, 1)
#     img_lr = transforms.ToPILImage()(img_lr)
#     img_lr.save(Path(f"{output_dir}") / (im.stem.split('_')[0] + f"_{scale}x.jpg"))
#     # pilimg = img_lr.resize((1024, 1024), Image.NEAREST)
#     # pilimg.save(Path(f"{output_dir}") / (im.stem.split('_')[0] + f"_LR_nn.jpg"))
