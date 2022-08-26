#!/bin/sh
#vals=$(seq 0 1 2)
for i in $(seq 0 1 79); do
    python test.py --img_idx $i --ckpt "checkpoint/stylegan2-ffhq-config-f.pt" --factor 64 --gpu_num 0
done