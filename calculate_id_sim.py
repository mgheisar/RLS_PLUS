import argparse
import math
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from PIL import Image
from glob import glob
from id_loss import IDLoss
from metrics.ms_ssim import MSSSIM
from lpips.lpips import LPIPS
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Images(Dataset):
    def __init__(self, image_list, duplicates):
        # args.files = [sorted(glob.glob(f"input/project/inputt/*_{args.factor}x.jpg"))[args.img_idx]]
        self.image_list = image_list
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        image = self.transform(Image.open(img_path)).to(device)
        return image, os.path.splitext(os.path.basename(img_path))[0]


def calculate_fid_folder(args):
    # args.input_dir = "input/project/resSR/32x_base_brgm"
    # args.input_dir = "input/project/resSR/RLSPlus/train/8x"
    image_list = sorted(glob(f"{args.input_dir}/*.jpg"))
    args.num_sample = min(args.num_sample, len(image_list))
    image_list = image_list[:args.num_sample]
    dataset_gen = Images(image_list, duplicates=1)
    # create dataloader
    data_loader_gen = DataLoader(
        dataset=dataset_gen,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=None,
        drop_last=False)

    image_list = sorted(glob(f"{args.gt_dir}/*.jpg"))
    image_list = image_list[:args.num_sample]
    dataset_gt = Images(image_list, duplicates=1)
    # dataset = build_dataset(opt)

    # create dataloader
    data_loader_gt = DataLoader(
        dataset=dataset_gt,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=None,
        drop_last=False)

    id_loss = IDLoss(args).to(device).eval()
    loss_func = MSSSIM().to(device).eval()
    lpips_loss = LPIPS(net_type='alex').to(device).eval()
    # features = extract_inception_features(data_generator(data_loader, total_batch), inception, total_batch, device)
    sim, msssim, lpipss = [], [], []
    for gt_im, gen_im in zip(data_loader_gt, data_loader_gen):
        gt_im = gt_im[0].to(device)
        gen_im = gen_im[0].to(device)
        # loss_id, sim_improvement = id_loss(gen_im, gt_im)
        loss_id, sim_improvement, id_logs = id_loss(gen_im, gt_im, gt_im)
        mssim_loss = loss_func(gen_im, gt_im)
        lpipss_loss = lpips_loss(gen_im, gt_im).mean()
        # print("id_logs", id_logs)
        # print("sim_id", 1-loss_id)
        # print("mssim_loss", mssim_loss)
        # print("lpips_loss", lpipss_loss)
        sim.append((1-loss_id).detach().cpu().numpy())
        msssim.append(mssim_loss.detach().cpu().numpy())
        lpipss.append(lpipss_loss.detach().cpu().numpy())
    print(f"Total id sim on {args.input_dir}: \n", np.mean(sim))
    print(f"Total msssim on {args.input_dir}: \n", np.mean(msssim))
    print(f"Total lpips on {args.input_dir}: \n", np.mean(lpipss))
    return np.mean(sim), np.mean(msssim), np.mean(lpipss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-restored_folder', type=str, help='Path to the folder.', required=True)
    parser.add_argument(
        '--fid_stats',
        type=str,
        help='Path to the dataset fid statistics.',
        default='input/project/fid_stats_hr.pth')  # input/project/fid_stats_hr.pth
    parser.add_argument('--input_dir', type=str, help='Path to the dataset.',
                        default='input/project/resSR/RLSPlus/train/16x')
    parser.add_argument('--gt_dir', type=str, help='Path to the dataset.', default='input/project/resHR')
    parser.add_argument('--batch_size', type=int, default=50)  # 64
    parser.add_argument('--num_sample', type=int, default=2000)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--backend', type=str, default='disk', help='io backend for dataset. Option: disk, lmdb')
    parser.add_argument('--ir_se50_weights', type=str, default='checkpoint/model_ir_se50.pth')
    args = parser.parse_args()
    dir_path = "input/project/resSR/RLSPlus/Test_ablation_pco"
    dirs = os.listdir(dir_path)
    sim, msssim, lpipss = [], [], []
    logp = []
    for dir in dirs:
        args.input_dir = os.path.join(dir_path, dir)
        print(dir)
        print('0.' + dir.split('logp')[1])
        logp.append('0.' + dir.split('logp')[1])
        sim_, msssim_, lpipss_ = calculate_fid_folder(args)
        sim.append(sim_)
        msssim.append(msssim_)
        lpipss.append(lpipss_)
    print("sim", sim)
    print("msssim", msssim)
    print("lpips", lpipss)
    print("logp", logp)
    import matplotlib.pyplot as plt
    # plot the figure to show the results in each subplots
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(logp, msssim, 'r')
    plt.xlabel(r'$\lambda_w$')
    plt.ylabel('MSSSIM')
    # plt.subplot(1, 3, 2)
    # plt.plot(logp, msssim, 'c')
    # plt.xlabel(r'$\lambda_w$')
    # plt.ylabel('msssim')
    plt.subplot(1, 2, 2)
    plt.plot(logp, lpipss, 'b')
    # print x label which is lambda_w with latex format
    plt.xlabel(r'$\lambda_w$')
    plt.ylabel('LPIPS')
    plt.savefig('input/project/resSR/RLSPlus/id_sim__Test_ablation.png')
