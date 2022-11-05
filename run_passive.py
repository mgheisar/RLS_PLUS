import os
import sys

ckpt = "checkpoint/stylegan2-ffhq-config-f.pt"
# factor = 8
# cmd = f'export CUDA_VISIBLE_DEVICES=0 && nohup {sys.executable} sr_boost.py --ckpt {ckpt} ' \
#       f'--factor {factor} --input_dir "input/project/resLR_{factor}x" --duplicate 1 ' \
#       f'--out_dir "input/project/resSR/{factor}x_base" > run_{factor}.out 2> run_{factor}.err &'
# os.system(cmd)
# ##-------------------------------------Ablation-------------------------------------
# out_dir = "input/project/resSR/ablation/"
# exp = "logp0_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=2 && nohup {sys.executable} sr_boost.py --ckpt {ckpt} ' \
#       f'--factor 64 --out_dir {os.path.join(out_dir, exp)} --logp 0' \
#       f' > run_{exp}.out 2> run_{exp}.err &'
# os.system(cmd)
#
# exp = "pnorm0_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} sr_boost.py --ckpt {ckpt} ' \
#       f'--factor 64 --out_dir {os.path.join(out_dir, exp)} --pnorm 0' \
#       f' > run_{exp}.out 2> run_{exp}.err &'
# os.system(cmd)
#
# exp = "cross0_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} sr_boost.py --ckpt {ckpt} ' \
#       f'--factor 64 --out_dir {os.path.join(out_dir, exp)} --cross 0' \
#       f' > run_{exp}.out 2> run_{exp}.err &'
# os.system(cmd)
#
# exp = "wFalse_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} sr_boost.py --ckpt {ckpt} ' \
#       f'--factor 64 --out_dir {os.path.join(out_dir, exp)} --w_plus' \
#       f' > run_{exp}.out 2> run_{exp}.err &'
# os.system(cmd)

# exp = "wo_regularization_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=3 && nohup {sys.executable} sr_boost.py --ckpt {ckpt} ' \
#       f'--factor 64 --out_dir {os.path.join(out_dir, exp)} --logp 0 --cross 0 --pnorm 0' \
#       f' > run_{exp}.out 2> run_{exp}.err &'
# os.system(cmd)
###--------------------Robustness--------------------------------------------------
# augs = "motionblur"
# cmd = f'export CUDA_VISIBLE_DEVICES=2 && nohup {sys.executable} sr_boost.py ' \
#       f'--ckpt {ckpt} --augs {augs} > run_{augs}.out 2> run_{augs}.err &'
# os.system(cmd)
#
# augs = "gaussiannoise"
# cmd = f'export CUDA_VISIBLE_DEVICES=2 && nohup {sys.executable} sr_boost.py ' \
#       f'--ckpt {ckpt} --augs {augs} > run_{augs}.out 2> run_{augs}.err &'
# os.system(cmd)
#
# augs = "saltpepper"
# cmd = f'export CUDA_VISIBLE_DEVICES=2 && nohup {sys.executable} sr_boost.py ' \
#       f'--ckpt {ckpt} --augs {augs} > run_{augs}.out 2> run_{augs}.err &'
# os.system(cmd)
#
# augs = "gaussianblur"
# cmd = f'export CUDA_VISIBLE_DEVICES=2 && nohup {sys.executable} sr_boost.py ' \
#       f'--ckpt {ckpt} --augs {augs} > run_{augs}.out 2> run_{augs}.err &'
# os.system(cmd)
# # #-------------------------------------Ablation FID-------------------------------------
in_dir = "input/project/resSR/ablation"
# exp = "cross0_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=0 && nohup {sys.executable} calculateFID.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_FID.out 2> run_{exp}_FID.err'
# os.system(cmd)

# exp = "logp0_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=0 && nohup {sys.executable} calculateFID.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_FID.out 2> run_{exp}_FID.err &'
# os.system(cmd)
#
# exp = "pnorm0_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=0 && nohup {sys.executable} calculateFID.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_FID.out 2> run_{exp}_FID.err &'
# os.system(cmd)

# exp = "logp0pnorm0cross0wTrue_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculateFID.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_FID.out 2> run_{exp}_FID.err &'
# os.system(cmd)
#
# exp = "logp0pnorm0cross0wFalse_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculateFID.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_FID.out 2> run_{exp}_FID.err &'
# os.system(cmd)
#
# exp = "wFalse_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=3 && nohup {sys.executable} calculateFID.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_FID.out 2> run_{exp}_FID.err &'
# os.system(cmd)

# ##-------------------------------------Ablation NIQE-------------------------------------
# in_dir = "input/project/resSR/ablation"
# exp = "cross0_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculateNIQE.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_NIQE.out 2> run_{exp}_NIQE.err &'
# os.system(cmd)
#
#
# exp = "logp0_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculateNIQE.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_NIQE.out 2> run_{exp}_NIQE.err &'
# os.system(cmd)
#
# exp = "pnorm0_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculateNIQE.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_NIQE.out 2> run_{exp}_NIQE.err &'
# os.system(cmd)
#
# exp = "logp0pnorm0cross0wFalse_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculateNIQE.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_NIQE.out 2> run_{exp}_NIQE.err &'
# os.system(cmd)
#
# exp = "wFalse_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculateNIQE.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_NIQE.out 2> run_{exp}_NIQE.err &'
# os.system(cmd)
#
# exp = "logp0pnorm0cross0wTrue_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculateNIQE.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_NIQE.out 2> run_{exp}_NIQE.err &'
# os.system(cmd)
# ##-------------------------------------Ablation PSNR SSIM-------------------------------------
# in_dir = "input/project/resSR/ablation"
# exp = "cross0_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculatePSNRSSIM.py ' \
#       f'--restored {os.path.join(in_dir, exp)} --gt "input/project/resHR" > run_{exp}_PSNR.out 2> run_{exp}_PSNR.err &'
# os.system(cmd)
#
#
# exp = "logp0_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculatePSNRSSIM.py ' \
#       f'--restored {os.path.join(in_dir, exp)} --gt "input/project/resHR" > run_{exp}_PSNR.out 2> run_{exp}_PSNR.err &'
# os.system(cmd)
#
# exp = "pnorm0_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculatePSNRSSIM.py ' \
#       f'--restored {os.path.join(in_dir, exp)} --gt "input/project/resHR" > run_{exp}_PSNR.out 2> run_{exp}_PSNR.err &'
# os.system(cmd)
#
# exp = "logp0pnorm0cross0wFalse_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculatePSNRSSIM.py ' \
#       f'--restored {os.path.join(in_dir, exp)} --gt "input/project/resHR" > run_{exp}_PSNR.out 2> run_{exp}_PSNR.err &'
# os.system(cmd)
#
# exp = "wFalse_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculatePSNRSSIM.py ' \
#       f'--restored {os.path.join(in_dir, exp)} --gt "input/project/resHR" > run_{exp}_PSNR.out 2> run_{exp}_PSNR.err &'
# os.system(cmd)
#
# exp = "logp0pnorm0cross0wTrue_64"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculatePSNRSSIM.py ' \
#       f'--restored {os.path.join(in_dir, exp)} --gt "input/project/resHR" > run_{exp}_PSNR.out 2> run_{exp}_PSNR.err &'
# os.system(cmd)
# #-------------------------------------Base scores-------------------------------------

# in_dir = "input/project/"
# exp = "resours"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculateNIQE.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_NIQE.out 2> run_{exp}_NIQE.err &'
# os.system(cmd)
#
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculateFID.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_FID.out 2> run_{exp}_FID.err &'
# os.system(cmd)
#
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculatePSNRSSIM.py ' \
#       f'--restored {os.path.join(in_dir, exp)} --gt "input/project/resHR" ' \
#       f'> run_{exp}_PSNR.out 2> run_{exp}_PSNR.err &'
# os.system(cmd)

# # #-------------------------------------Robustness FID-------------------------------------
# in_dir = "input/project/resSR"
# exp = "gaussiannoise"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculateNIQE.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_NIQE.out 2> run_{exp}_NIQE.err &'
# os.system(cmd)
#
# exp = "saltpepper"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculateNIQE.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_NIQE.out 2> run_{exp}_NIQE.err &'
# os.system(cmd)
#
# exp = "gaussianblur"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculateNIQE.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_NIQE.out 2> run_{exp}_NIQE.err &'
# os.system(cmd)
#
# exp = "motionblur"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculateNIQE.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_NIQE.out 2> run_{exp}_NIQE.err &'
# os.system(cmd)
#
# exp = "motionblur_100_1"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculateNIQE.py ' \
#       f'--input_dir {os.path.join(in_dir, exp)} > run_{exp}_NIQE.out 2> run_{exp}_NIQE.err &'
# os.system(cmd)

# #-------------------------------------Robustness PSNR SSIM-------------------------------------
# in_dir = "input/project/resSR"
# exp = "gaussiannoise"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculatePSNRSSIM.py ' \
#       f'--restored {os.path.join(in_dir, exp)} --gt "input/project/resHR"
#       > run_{exp}_PSNR.out 2> run_{exp}_PSNR.err &'
# os.system(cmd)
#
# exp = "saltpepper"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculatePSNRSSIM.py ' \
#       f'--restored {os.path.join(in_dir, exp)} --gt "input/project/resHR"
#       > run_{exp}_PSNR.out 2> run_{exp}_PSNR.err &'
# os.system(cmd)
#
# exp = "gaussianblur"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculatePSNRSSIM.py ' \
#       f'--restored {os.path.join(in_dir, exp)} --gt "input/project/resHR"
#       > run_{exp}_PSNR.out 2> run_{exp}_PSNR.err &'
# os.system(cmd)
#
# exp = "motionblur"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculatePSNRSSIM.py ' \
#       f'--restored {os.path.join(in_dir, exp)} --gt "input/project/resHR"
#       > run_{exp}_PSNR.out 2> run_{exp}_PSNR.err &'
# os.system(cmd)
#
# exp = "motionblur_100_1"
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} calculatePSNRSSIM.py ' \
#       f'--restored {os.path.join(in_dir, exp)} --gt "input/project/resHR"
#       > run_{exp}_PSNR.out 2> run_{exp}_PSNR.err &'
# os.system(cmd)

# cmd = f'nohup {sys.executable} sr_boost_ada.py > run_ada.out 2> run_ada.err &'
# os.system(cmd)

n = 8
r = 1
factor = 32
# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} test_mean.py ' \
#       f'--ckpt {ckpt} --steps 500' \
#       f' --factor {factor} --input_dir "input/project/resLR_{factor}x" --lr 0.5 ' \
#       f'--out_dir "input/project/resSR/test/train"' \
#       f' > run_1.out 2> run_1.err &'
# os.system(cmd)

cmd = f'export CUDA_VISIBLE_DEVICES=0 && nohup {sys.executable} temp.py ' \
      f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps 41' \
      f' --factor {factor} --input_dir "input/project/resLR_{factor}x" --lr 0.1 ' \
      f'--out_dir "input/project/resSR/test/train/{factor}x_test"' \
      f' > run_{n}_{r}_{factor}.out 2> run_{n}_{r}_{factor}.err &'
os.system(cmd)

cmd = f'export CUDA_VISIBLE_DEVICES=0 && nohup {sys.executable} temp.py ' \
      f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps 41' \
      f' --factor {factor} --input_dir "input/project/resLR_{factor}x" --lr 0.1 ' \
      f'--out_dir "input/project/resSR/test/train/{factor}x_test"' \
      f' > run_{n}_{r}_{factor}.out 2> run_{n}_{r}_{factor}.err &'
os.system(cmd)

# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} temp_wn.py ' \
#       f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps 121 ' \
#       f'--factor {factor} --input_dir "input/project/resLR_{factor}x" --lr 0.5 ' \
#       f'--out_dir "input/project/resSR/test/train/{factor}x"' \
#       f' > runwn_{n}_{r}_{factor}.out 2> runwn_{n}_{r}_{factor}.err &'
# os.system(cmd)

# cmd = f'export CUDA_VISIBLE_DEVICES=1 && nohup {sys.executable} temp_g.py ' \
#       f'--ckpt {ckpt} --num_trainable_noise_layers {n} --radius {r} --steps 71 ' \
#       f'--factor {factor} --input_dir "input/project/resLR_{factor}x" --lr 0.5 ' \
#       f'--out_dir "input/project/resSR/test/train/{factor}x" ' \
#       f' >rung_{n}_{r}_{factor}.out 2> rung_{n}_{r}_{factor}.err &'
# os.system(cmd)