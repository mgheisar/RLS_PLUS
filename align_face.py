import random
from glob import glob
import scipy
import scipy.ndimage
import dlib
from drive import open_url
from pathlib import Path
import argparse
from bicubic import BicubicDownSample
import torchvision
from shape_predictor import align_face

parser = argparse.ArgumentParser(description='PULSE')

parser.add_argument('-input_dir', type=str, default='input/project/LR_team', help='directory with unprocessed images')
parser.add_argument('-output_dir', type=str, default='input/project/', help='output directory')
parser.add_argument('-factor', type=int, default=32, help='scale to downscale the input images to, must be power of 2')
parser.add_argument('-seed', type=int, help='manual seed to use')
parser.add_argument('-cache_dir', type=str, default='cache', help='cache directory for model weights')

args = parser.parse_args()
hr_size = 1024
cache_dir = Path(args.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading Shape Predictor")
f = open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir=cache_dir, return_path=True)
predictor = dlib.shape_predictor(f)
# files = glob(f"{Path(args.input_dir)}/*.jpg")
# random.shuffle(files)
# files = files[:240]
files = Path(args.input_dir).glob("*.jpg")
ii = 0
for im in files:
    # png = PIL.Image.open(im)
    # png.load()  # required for png.split()
    # background = PIL.Image.new("RGB", png.size, (255, 255, 255))
    # background.paste(png, mask=png.split()[3])  # 3 is the alpha channel
    # background.save('foo.jpg', 'JPEG', quality=1024)
    # exit(0)
    faces = align_face(str(im), predictor, hr_size)
    for i,face in enumerate(faces):
        # face.save(Path(args.output_dir) / (im.stem + f"_HR.jpg"))
        if(args.factor):
            D = BicubicDownSample(factor=args.factor)
            face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
            face_tensor_lr = D(face_tensor)[0].cpu().detach().clamp(0, 1)
            face = torchvision.transforms.ToPILImage()(face_tensor_lr)

        face.save(Path(args.output_dir) / (im.stem.split('_')[0]+f"_{args.factor}x.jpg"))
    ii = ii + 1
    if ii == 1020:
        exit(0)