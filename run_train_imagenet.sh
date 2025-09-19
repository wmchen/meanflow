#! /bin/bash

cd /home/chenwm/meanflow

output_dirname="DiT-XL-2_256"

export CUDA_VISIBLE_DEVICES=0,1

nohup accelerate launch --num_processes=2 train_imagenet.py \
    --output_dir "result/$output_dirname" \
    --tracker_project_name "MeanFlow_DiT-XL/2" \
    --data_path "/home/ailab/datasets_nas/ImageNet-1k/train" \
    --image_size 256 \
    --num_classes 1000 \
    --seed 42 \
    --model "DiT-XL/2" \
    --vae "/home/ailab/model_weights_nas/stable-diffusion/vae/sd-vae-ft-mse" \
    --timestep_equal_ratio 0.75 \
    --cfg_ratio 1.0 \
    --cfg_w_prime 2.5 \
    --cfg_w 0.2 \
    --cfg_trigger_t_min 0.3 \
    --cfg_trigger_t_max 0.8 \
    --p 1.0 \
    --eps 0.000001 \
    --num_train_epochs 80 \
    --train_batch_size 16 \
    --dataloader_num_workers 0 \
    --gradient_accumulation_steps 1 \
    --use_ema \
    --foreach_ema \
    --learning_rate 0.0001 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_weight_decay 0.01 \
    --adam_epsilon 0.00000001 \
    --max_grad_norm 1.0 \
    --lr_scheduler "constant" \
    --lr_warmup_steps 500 \
    --checkpointing_steps 10 \
    --checkpoints_total_limit 10 \
    --val_inference_steps 1 2 4 \
    --val_class_labels 207 360 387 974 88 979 417 279 \
    --val_images_each_row 4 \
    > log_$output_dirname.txt 2>&1 &
