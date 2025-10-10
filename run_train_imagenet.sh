#! /bin/bash
#BSUB -J wmchen
#BSUB -q gpu-eee-hezh
#BSUB -n 16
#BSUB -R "span[ptile=16]"
#BSUB -e /work/eee-chenwm/Documents/meanflow/log_lsf/err/%J.err
#BSUB -o /work/eee-chenwm/Documents/meanflow/log_lsf/out/%J.out
#BSUB -gpu "num=4"
#BSUB -m "b03u04g"

source ~/.bashrc
source activate meanflow

cd /work/eee-chenwm/Documents/meanflow

output_dirname="20251010_MF_DiT-B-4_256"

accelerate launch \
    --num_processes=8 \
    --num_machines=2 \
    --machine_rank=1 \
    --main_process_ip="b03u02g" \
    train_imagenet.py \
    --output_dir "result/$output_dirname" \
    --tracker_project_name "MeanFlow" \
    --data_path "/work/eee-chenwm/Documents/datasets/ImageNet-1k/train" \
    --image_size 256 \
    --num_classes 1000 \
    --seed 42 \
    --model "DiT-B/4" \
    --vae "/work/eee-chenwm/Documents/checkpoints/vae/sd-vae-ft-mse" \
    --timestep_equal_ratio 0.75 \
    --timestep_dist_type "lognorm" \
    --timestep_dist_mu -0.4 \
    --timestep_dist_sigma 1.0 \
    --cfg \
    --cfg_w 3.0 \
    --cfg_k 0.0 \
    --cfg_trigger_t_min 0.0 \
    --cfg_trigger_t_max 1.0 \
    --cfg_cond_dropout 0.1 \
    --p 1.0 \
    --eps 0.000001 \
    --num_train_epochs 80 \
    --train_batch_size 32 \
    --dataloader_num_workers 16 \
    --gradient_accumulation_steps 1 \
    --use_ema \
    --ema_decay 0.9999 \
    --learning_rate 0.0001 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_weight_decay 0.0 \
    --adam_epsilon 0.00000001 \
    --max_grad_norm 1.0 \
    --lr_scheduler "constant" \
    --lr_warmup_steps 0 \
    --checkpointing_steps 10000 \
    --checkpoints_total_limit 10 \
    --val_inference_steps 1 2 \
    --val_class_labels 207 360 387 974 88 979 417 279 145 0 \
    --val_images_each_row 5 \
    --remarks "Fixed latent positional embedding. Timestep positional embedding: (t, t-r)"
