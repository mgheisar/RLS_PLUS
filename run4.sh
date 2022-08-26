#!/bin/sh
#vals=$(seq 0 1 2)
for i in $(seq 160 1 239); do
    python test.py --img_idx $i --ckpt "checkpoint/stylegan2-ffhq-config-f.pt" --factor 32 --gpu_num 2
done