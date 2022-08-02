#!/bin/sh
lr_vec=( 0.01 0.05 0.1 0.25 0.5)
logp_vec=( 0.0001 0.0005 0.001 0.005 0.01)
cross_vec=( 0.01 0.05 0.1 0.5)
pnorm_vec=( 0 0.001 0.01 0.1)


for fp1 in "${lr_vec[@]}"
do
  for fp2 in "${logp_vec[@]}"
  do
    for fp3 in "${cross_vec[@]}"
    do
        for fp4 in "${pnorm_vec[@]}"
        do
          python temp2.py --steps 500 --lr ${fp1/,/.} --logp ${fp2/,/.} --cross ${fp3/,/.} \
          --pnorm ${fp4/,/.} --ckpt "checkpoint/face256.pt" --factor 32 --out_dir "/ffhq1_8"  "input/ffhq1_8.png"
        done
    done
  done
done