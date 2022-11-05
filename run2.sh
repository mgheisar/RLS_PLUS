#!/bin/sh
#vals=$(seq 0 1 2)
for i in $(seq 5 1 13); do
  for j in $(seq 1 1 5); do
      python temp.py --ckpt "checkpoint/stylegan2-ffhq-config-f.pt" --factor 16 --input_dir "input/project/lrr/lr_16" --num_trainable_noise_layers $i --radius j
  done
done