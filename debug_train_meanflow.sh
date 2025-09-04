#! /bin/bash

cd /root/DiT

output_dirname="test"

export CUDA_VISIBLE_DEVICES=0,1
torchrun --nnodes=1 --nproc_per_node=1 train_meanflow.py \
    --data-path "/data/ImageNet-1k/train" \
    --results-dir "/data/results/MeanFlow/$output_dirname" \
    --model "DiT-XL/2" \
    --vae "/data/sd-vae-ft-mse" \
    --image-size 256 \
    --num-classes 1000 \
    --epochs 1000 \
    --global-batch-size 16 \
    --global-seed 42 \
    --lr 0.0001 \
    --num-workers 4 \
    --log-every 1 \
    --ckpt-every 10000 \
    --cfg_w_prime 2.5 \
    --cfg_w 0.2 \
    --cfg_trigger_t_min 0.3 \
    --cfg_trigger_t_max 0.8
