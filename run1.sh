#!/bin/sh
lr_vec=( 0.1 0.5)
logp_vec=( 0.0001 0.0005 0.001 0.005 0.01)
cross_vec=( 0.01 0.05 0.1 0.5)
pnorm_vec=( 0 0.001 0.01 0.1)
input_vec1=( "input/biden1_32.png" "input/obama_32.png")5
input_vec2=( "input/biden1_8.png" "input/obama_8.png")
#for fp1 in "${lr_vec[@]}"
#do
#  for fp2 in "${logp_vec[@]}"
#  do
#    for fp3 in "${cross_vec[@]}"
#    do
#        for fp4 in "${pnorm_vec[@]}"
#        do
#          python sr_boost.py --steps 500 --lr ${fp1/,/.} --logp ${fp2/,/.} --cross ${fp3/,/.} \
#          --pnorm ${fp4/,/.} --ckpt "checkpoint/face256.pt" --factor 8 --out_dir "/obama_32"  "input/obama_32.png"
#        done
#    done
#  done
#done

for fp4 in "${input_vec1[@]}"
do
  python temp2.py --ckpt "checkpoint/face256.pt" --factor 8 ${fp4}
done

#nohup python temp2.py --ckpt "checkpoint/face256.pt" --gpu_num 0 --factor 16 --out_dir "/00006_16x"  "input/00006_16x.png" > run_16x.out &
#nohup python temp3.py --ckpt "checkpoint/face256.pt" --gpu_num 0 --factor 16 --out_dir "/16x" > run_16x.out &


nohup python temp2.py --ckpt "checkpoint/stylegan2-ffhq-config-f.pt" --gpu_num 1 --factor 64 --out_dir "/00006_64x"  --files "input/00006_64x.png" > run_00006_64x.out &

nohup python temp2.py --ckpt "checkpoint/stylegan2-ffhq-config-f.pt" --gpu_num 0 --factor 64 --out_dir "/00420_64x"  --files "input/00420_64x.png" > run_00420_64x.out &