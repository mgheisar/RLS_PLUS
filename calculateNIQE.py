import argparse
import cv2
import os
import warnings
from glob import glob

from metrics import calculate_niqe


def main(args):

    niqe_all = []
    imgs = sorted(glob(os.path.join(args.input_dir, '*.jpg')))[:args.num_samples]
    for i, img_path in enumerate(imgs):
        basename, _ = os.path.splitext(os.path.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            niqe_score = calculate_niqe(img, args.crop_border, input_order='HWC', convert_to='y')
        print(f'{i+1:3d}: {basename:25}. \tNIQE: {niqe_score:.6f}')
        niqe_all.append(niqe_score)

    print(args.input_dir)
    print(f'Average: NIQE: {sum(niqe_all) / len(niqe_all):.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='input/project/resHR', help='Input path')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    parser.add_argument('--num_samples', type=int, default=2000, help='Number of samples')
    args = parser.parse_args()
    main(args)