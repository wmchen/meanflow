#! /bin/bash

cd /home/chenwm/meanflow

export CUDA_VISIBLE_DEVICES=1

python sample.py \
    --model_name "DiT-XL/2" \
    --model_path "log/XL_0050000.pt" \
    --vae "/home/ailab/model_weights_nas/stable-diffusion/vae/sd-vae-ft-mse" \
    --class_labels 207 360 387 974 88 979 417 279 \
    --image-size 256 \
    --num-classes 1000 \
    --num_steps 1 \
    --guidance_scale 1.0 \
    --num_images_each_row 4 \
    --output_path "sample.png"
