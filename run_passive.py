import os
import sys

ckpt = "checkpoint/stylegan2-ffhq-config-f.pt"
factor = 64
# cmd = f'export CUDA_VISIBLE_DEVICES=2 && nohup {sys.executable} pulse_sr.py --ckpt {ckpt} ' \
#       f'--factor {factor} --input_dir "input/project/resLR_{factor}x" --duplicate 1 ' \
#       f'--out_dir "input/project/resSR/{factor}x_base_pulse" > run_{factor}_pulse.out 2> run_{factor}_pulse.err &'
# os.system(cmd)
#
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} BRGM_adapted.py --ckpt {ckpt} ' \
#       f'--factor {factor} --input_dir "input/project/resLR_{factor}x" --duplicate 1 ' \
#       f'--out_dir "input/project/resSR/{factor}x_base_brgm" > run_{factor}_brgm.out 2> run_{factor}_brgm.err &'
# os.system(cmd)

# ##-------------------------------------Ablation-------------------------------------
# factor = 32
# out_dir = "input/project/resSR/ablation_RLS/"
# exp = "logp0_32"
# cmd = f'export CUDA_VISIBLE_DEVICES=2 && nohup {sys.executable} sr_boost.py --ckpt {ckpt} ' \
#       f'--factor {factor} --out_dir {os.path.join(out_dir, exp)} --logp 0 --save_anchor ' \
#       f'--input_dir "input/project/resLR_{factor}x" > run_{exp}.out 2> run_{exp}.err &'
# os.system(cmd)
#
# exp = "pnorm0_32"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} sr_boost.py --ckpt {ckpt} ' \
#       f'--factor {factor}  --out_dir {os.path.join(out_dir, exp)} --pnorm 0 --save_anchor ' \
#       f'--input_dir "input/project/resLR_{factor}x" > run_{exp}.out 2> run_{exp}.err &'
# os.system(cmd)
#
# exp = "cross0_32"
# cmd = f'export CUDA_VISIBLE_DEVICES=2 && nohup {sys.executable} sr_boost.py --ckpt {ckpt} ' \
#       f'--factor {factor}  --out_dir {os.path.join(out_dir, exp)} --cross 0 --save_anchor ' \
#       f'--input_dir "input/project/resLR_{factor}x" > run_{exp}.out 2> run_{exp}.err &'
# os.system(cmd)
#
# exp = "wFalse_32"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} sr_boost.py --ckpt {ckpt} ' \
#       f'--factor {factor}  --out_dir {os.path.join(out_dir, exp)} --w_plus --save_anchor ' \
#       f'--input_dir "input/project/resLR_{factor}x" > run_{exp}.out 2> run_{exp}.err &'
# os.system(cmd)
#
# exp = "wo_regularization_32"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} sr_boost.py --ckpt {ckpt} ' \
#       f'--factor {factor}  --out_dir {os.path.join(out_dir, exp)} --logp 0 --cross 0 --pnorm 0 --save_anchor ' \
#       f'--input_dir "input/project/resLR_{factor}x" > run_{exp}.out 2> run_{exp}.err &'
# os.system(cmd)
#  ##--------------------Robustness--------------------------------------------------
# factor, steps_ada = 16, 70
# n = 9
# r = 1
# augs = "motionblur"
# out_dir = "input/project/resSR/robustness_RLSPlus"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} temp.py ' \
#       f'--ckpt {ckpt} --factor {factor} --input_dir "input/project/resLR_{factor}x" ' \
#       f'--num_trainable_noise_layers_ada {n} --radius_ada {r} --steps_ada {steps_ada} --lr_ada 0.1 ' \
#       f'--out_dir {out_dir} --augs {augs} > run_{augs}.out 2> run_{augs}.err &'
# os.system(cmd)
#
# augs = "gaussiannoise"
# out_dir = "input/project/resSR/robustness_RLSPlus"
# cmd = f'export CUDA_VISIBLE_DEVICES=2 && nohup {sys.executable} temp.py ' \
#       f'--ckpt {ckpt} --factor {factor} --input_dir "input/project/resLR_{factor}x" ' \
#       f'--num_trainable_noise_layers_ada {n} --radius_ada {r} --steps_ada {steps_ada} --lr_ada 0.1 ' \
#       f'--out_dir {out_dir} --augs {augs} > run_{augs}.out 2> run_{augs}.err &'
# os.system(cmd)

# augs = "saltpepper"
# out_dir = "input/project/resSR/robustness_RLSPlus"
# cmd = f'export CUDA_VISIBLE_DEVICES=0 && nohup {sys.executable} temp.py ' \
#       f'--ckpt {ckpt} --factor {factor} --input_dir "input/project/resLR_{factor}x" ' \
#       f'--num_trainable_noise_layers_ada {n} --radius_ada {r} --steps_ada {steps_ada} --lr_ada 0.1 ' \
#       f'--out_dir {out_dir} --augs {augs} > run_{augs}.out 2> run_{augs}.err &'
# os.system(cmd)
#
# augs = "gaussianblur"
# out_dir = "input/project/resSR/robustness_RLSPlus"
# cmd = f'export CUDA_VISIBLE_DEVICES=3 && nohup {sys.executable} temp.py ' \
#       f'--ckpt {ckpt} --factor {factor} --input_dir "input/project/resLR_{factor}x" ' \
#       f'--num_trainable_noise_layers_ada {n} --radius_ada {r} --steps_ada {steps_ada} --lr_ada 0.1 ' \
#       f'--out_dir {out_dir} --augs {augs} > run_{augs}.out 2> run_{augs}.err &'
# os.system(cmd)

# # factor, steps = 8, 80
# # factor, steps = 16, 70
# factor, steps = 32, 30
# augs = "motionblur"
# out_dir = "input/project/resSR/robustness"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} sr_boost.py ' \
#       f'--ckpt {ckpt} --factor {factor} --input_dir "input/project/resLR_{factor}x" ' \
#       f'--out_dir {out_dir} --augs {augs} > run_{augs}.out 2> run_{augs}.err &'
# os.system(cmd)
#
# augs = "gaussiannoise"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} sr_boost.py ' \
#       f'--ckpt {ckpt} --factor {factor} --input_dir "input/project/resLR_{factor}x" ' \
#       f'--out_dir {out_dir} --augs {augs} > run_{augs}.out 2> run_{augs}.err &'
# os.system(cmd)
#
# augs = "saltpepper"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} sr_boost.py ' \
#       f'--ckpt {ckpt} --factor {factor} --input_dir "input/project/resLR_{factor}x" ' \
#       f'--out_dir {out_dir} --augs {augs} > run_{augs}.out 2> run_{augs}.err &'
# os.system(cmd)
#
# augs = "gaussianblur"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} sr_boost.py ' \
#       f'--ckpt {ckpt} --factor {factor} --input_dir "input/project/resLR_{factor}x" ' \
#       f'--out_dir {out_dir} --augs {augs} > run_{augs}.out 2> run_{augs}.err'
# os.system(cmd)


# n = 9
# r = 1
# out_dir = "input/project/resSR/robustness"
# augs = "motionblur"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} sr_boost_ada.py ' \
#       f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps {steps} ' \
#       f'--factor {factor} --augs {augs} --input_dir "input/project/resLR_{factor}x" --lr 0.1 ' \
#       f'--out_dir {out_dir} > run_{augs}.out 2> run_{augs}.err &'
# os.system(cmd)
#
# augs = "gaussiannoise"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} sr_boost_ada.py ' \
#       f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps {steps} ' \
#       f'--factor {factor} --augs {augs} --input_dir "input/project/resLR_{factor}x" --lr 0.1 ' \
#       f'--out_dir {out_dir} > run_{augs}.out 2> run_{augs}.err &'
# os.system(cmd)
#
# augs = "saltpepper"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} sr_boost_ada.py ' \
#       f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps {steps} ' \
#       f'--factor {factor} --augs {augs} --input_dir "input/project/resLR_{factor}x" --lr 0.1 ' \
#       f'--out_dir {out_dir} > run_{augs}.out 2> run_{augs}.err &'
# os.system(cmd)
#
# augs = "gaussianblur"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} sr_boost_ada.py ' \
#       f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps {steps} ' \
#       f'--factor {factor} --augs {augs} --input_dir "input/project/resLR_{factor}x" --lr 0.1 ' \
#       f'--out_dir {out_dir} > run_{augs}.out 2> run_{augs}.err &'
# os.system(cmd)
# #-------------------------------------Scores-------------------------------------
# in_dir = "input/project/resSR/ablation_RLS/"
# exp = "cross0_32"
# exp = "logp0_32"
# exp = "pnorm0_32"
# exp = "wFalse_32"
# exp = "wo_regularization_32"
# in_dir = "input/project/resSR"
# exp = "32x_base"
# in_dir = "input/project/resSR/robustness_RLSPlus"
# # exp = "saltpepper"
# # exp = "gaussianblur"
# # exp = "gaussiannoise"
# exp = "motionblur"
# # in_dir = "input/project/resSR/RLSPlus/train"
# # exp = "16x"
#
# cmd = f'export CUDA_VISIBLE_DEVICES=2 && nohup {sys.executable} calculate_id_sim.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_idsim.out 2> run_{exp}_idsim.err &'
# os.system(cmd)
#
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} realistic_metrics.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_FID.out 2> run_{exp}_FID.err &'
# os.system(cmd)
#
# cmd = f'nohup {sys.executable} calculateNIQE.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_NIQE.out 2> run_{exp}_NIQE.err &'
# os.system(cmd)
#
# cmd = f'nohup {sys.executable} calculatePSNRSSIM.py ' \
#       f'--restored {os.path.join(in_dir, exp)} --gt "input/project/resHR" ' \
#       f'> run_{exp}_PSNR.out 2> run_{exp}_PSNR.err &'
# os.system(cmd)

# cmd = f'export CUDA_VISIBLE_DEVICES=0 && nohup {sys.executable} calculateFID.py ' \
#      f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_FID1.out 2> run_{exp}_FID1.err &'
# os.system(cmd)
# # #-------------------------------------RLS Plus-------------------------------------
n = 9
r = 1
# factor, steps = 8, 80
factor, steps = 16, 70
# factor, steps = 32, 30
# factor, steps = 64, 20

# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} temp_gn.py ' \
#       f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps {steps}' \
#       f' --factor {factor} --input_dir "input/project/resLR_{factor}x" --lr 0.1 ' \
#       f'--out_dir "input/project/resSR/RLSPlus/train/ablation/train_ng_anchor/{factor}x"' \
#       f' > run_gn_{factor}.out 2> run_gn_{factor}.err &'
# os.system(cmd)

# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} temp_wn.py ' \
#       f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps {steps}' \
#       f' --factor {factor} --input_dir "input/project/resLR_{factor}x" --lr 0.1 ' \
#       f'--out_dir "input/project/resSR/RLSPlus/train/ablation/train_wn_anchor/{factor}x"' \
#       f' > run_wn_{factor}.out 2> run_wn_{factor}.err &'
# os.system(cmd)
#
# cmd = f'export CUDA_VISIBLE_DEVICES=0 && nohup {sys.executable} temp_wng_mean.py ' \
#       f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps {steps}' \
#       f' --factor {factor} --input_dir "input/project/resLR_{factor}x" --lr 0.1 ' \
#       f'--out_dir "input/project/resSR/RLSPlus/train/ablation/train_wng_mean/{factor}x"' \
#       f' > run_wng_mean_{factor}.out 2> run_wng_mean_{factor}.err &'
# os.system(cmd)

# n = 0
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} sr_boost_ada.py ' \
#       f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps {steps}' \
#       f' --factor {factor} --input_dir "input/project/resLR_{factor}x" --lr 0.1 ' \
#       f'--out_dir "input/project/resSR/RLSPlus/train/ablation/train_wg_anchor/{factor}x"' \
#       f' > run_gw_{factor}.out 2> run_gw_{factor}.err &'
# os.system(cmd)

# n = 9
# cmd = f'export CUDA_VISIBLE_DEVICES=2 && nohup {sys.executable} sr_boost_ada.py ' \
#       f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps {steps}' \
#       f' --factor {factor} --input_dir "input/project/resLR_{factor}x" --lr 0.1 ' \
#       f'--out_dir "input/project/resSR/RLSPlus/train/ablation/train_ours_without_l1ball_constraint/{factor}x"' \
#       f' > run_l1ball_{factor}.out 2> run_l1ball_{factor}.err &'
# os.system(cmd)

# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} sr_boost_ada.py ' \
#       f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps {steps}' \
#       f' --factor {factor} --input_dir "input/project/resLR_{factor}x" --lr 0.1 ' \
#       f'--out_dir "input/project/resSR/RLSPlus/train/{factor}x"' \
#       f' > run_{n}_{r}_{factor}.out 2> run_{n}_{r}_{factor}.err &'
# os.system(cmd)

# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} test_mean.py ' \
#       f'--ckpt {ckpt} --steps 500' \
#       f' --factor {factor} --input_dir "input/project/resLR_{factor}x" --lr 0.5 ' \
#       f'--out_dir "input/project/resSR/RLSPlus/train"' \
#       f' > run_1.out 2> run_1.err &'
# os.system(cmd)

# cmd = f'export CUDA_VISIBLE_DEVICES=3 && nohup {sys.executable} sr_boost_ada_false.py ' \
#       f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps 41' \
#       f' --factor {factor} --input_dir "input/project/resLR_{factor}x" --lr 0.1 ' \
#       f'--out_dir "input/project/resSR/RLSPlus/train/{factor}x"' \
#       f' > run_{n}_{r}_{factor}_.out 2> run_{n}_{r}_{factor}.err_ &'
# os.system(cmd)

# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} temp_wn.py ' \
#       f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps 121 ' \
#       f'--factor {factor} --input_dir "input/project/resLR_{factor}x" --lr 0.1 ' \
#       f'--out_dir "input/project/resSR/RLSPlus/train/{factor}x"' \
#       f' > runwn_{n}_{r}_{factor}.out 2> runwn_{n}_{r}_{factor}.err &'
# os.system(cmd)

# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} temp_g.py ' \
#       f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps 71 ' \
#       f'--factor {factor} --input_dir "input/project/resLR_{factor}x" --lr 0.1 ' \
#       f'--out_dir "input/project/resSR/RLSPlus/train/{factor}x" ' \
#       f' >rung_{n}_{r}_{factor}.out 2> rung_{n}_{r}_{factor}.err &'
# os.system(cmd)
##


# import shutil
# from glob import glob
# img_list = sorted(glob(f"input/project/resHR/*.jpg"))[:150]
# for img in img_list:
#     shutil.copy(img, 'input/project/resHR_150/')
# exit(0)
# factor, steps_ada = 8, 40
# factor, steps_ada = 16, 70
# factor, steps_ada = 32, 30
# factor, steps_ada = 64, 20
# n = 9
# r = 1
# import dlib
# from drive import open_url
# from pathlib import Path
# from bicubic import BicubicDownSample
# import torchvision
# from shape_predictor import align_face
#
# file_dir = "input/project/HR_team"
# hr_size = 1024
# logp = 0  # 0.0001
# pnorm = 0.005
# cross = 0.05
# factor, steps_ada = 4, 50
# out_dir = "input/project/team4"
# cache_dir = Path('cache')
# cache_dir.mkdir(parents=True, exist_ok=True)
#
# input_dir = Path('input/project/LR_team')
# input_dir.mkdir(parents=True, exist_ok=True)
# print("Downloading Shape Predictor")
# f = open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir=cache_dir, return_path=True)
# predictor = dlib.shape_predictor(f)
# # load all images with jpg or png extension
# files = sorted(Path(file_dir).glob("*.jpg")) + sorted(Path(file_dir).glob("*.png")) + \
#         sorted(Path(file_dir).glob("*.jpeg"))
# for im in files:
#     faces = align_face(str(im), predictor, hr_size)
#     for i, face in enumerate(faces):
#         D = BicubicDownSample(factor=factor)
#         face_tensor = torchvision.transforms.ToTensor()(face).unsqueeze(0).cuda()
#         face_tensor_lr = D(face_tensor)[0].cpu().detach().clamp(0, 1)
#         face = torchvision.transforms.ToPILImage()(face_tensor_lr)
#         face.save(Path(input_dir) / (im.stem.split('_')[0] + f"_{factor}x.jpg"))

# import cv2
# img = cv2.imread("/projects/superres/Marzieh/RLS/input/project/Yusef/patch21.jpg")
# img = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_NEAREST)
# cv2.imwrite("/projects/superres/Marzieh/RLS/input/project/Yusef/patch21_NN.jpg", img)


# pnorm_vec = [0.0001, 0.0005, 0.001, 0.002, 0.005]
# cross_vec = [0.01, 0.05, 0.1, 0.5]
# for pnorm in pnorm_vec:
#       for cross in cross_vec:
#             cmd = f'export CUDA_VISIBLE_DEVICES=2 && nohup {sys.executable} temp.py ' \
#                   f'--ckpt {ckpt} --factor {factor} --input_dir {input_dir} ' \
#                   f'--steps 250 --lr 0.4 --num_trainable_noise_layers 5 --logp {logp} --cross {cross} ' \
#                   f'--pnorm {pnorm} --num_trainable_noise_layers_ada {n} ' \
#                   f'--radius_ada {r} --steps_ada {steps_ada} --lr_ada 0.1 --duplicate 5 ' \
#                   f'--out_dir {out_dir}'y
#             os.system(cmd)

# cmd = f'export CUDA_VISIBLE_DEVICES=2 && nohup {sys.executable} temp.py ' \
#       f'--ckpt {ckpt} --factor {factor} --input_dir {input_dir} ' \
#       f'--steps 250 --lr 0.4 --num_trainable_noise_layers 5 --logp {logp} --cross {cross} --pnorm {pnorm} ' \
#       f'--num_trainable_noise_layers_ada {n} ' \
#       f'--radius_ada {r} --steps_ada {steps_ada} --lr_ada 0.1 --duplicate 1 ' \
#       f'--out_dir {out_dir}'
# os.system(cmd)

# ckpt = "checkpoint/stylegan2-ffhq-config-f.pt"
# factor = 32
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} pulse_sr.py --ckpt {ckpt} ' \
#       f'--factor {factor} --w_plus --input_dir {input_dir} --duplicate 5 ' \
#       f'--out_dir {out_dir}'
# os.system(cmd)
# print(cmd)

ckpt = "checkpoint/stylegan2-ffhq-config-f.pt"
n = 9
r = 1
factor, steps_ada = 16, 70
logp = 0.05
out_dir = "input/project/resSR/RLSPlus/Test_ablation_pco/logp05"
gpu_num = 1
pnorm = 0  # 0.004
cross = 0  # 0.5
cmd = f'export CUDA_VISIBLE_DEVICES={gpu_num} && nohup {sys.executable} temp.py ' \
      f'--ckpt {ckpt} --factor {factor} --input_dir "input/project/resLR_{factor}x" ' \
      f'--steps 250 --lr 0.4 --logp {logp} --cross {cross} --pnorm {pnorm} ' \
      f'--num_trainable_noise_layers_ada {n} ' \
      f'--radius_ada {r} --steps_ada {steps_ada} --lr_ada 0.1 --duplicate 1 ' \
      f'--out_dir {out_dir}'
os.system(cmd)

