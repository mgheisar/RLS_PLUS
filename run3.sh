#!/bin/sh
#vals=$(seq 0 1 2)
for i in $(seq 80 1 159); do
    python test.py --img_idx $i --ckpt "checkpoint/stylegan2-ffhq-config-f.pt" --factor 32 --gpu_num 1
done