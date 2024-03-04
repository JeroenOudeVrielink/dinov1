#!/usr/bin/env bash

# Num workers was 14 (optimal) but caused OOM errors for RAM on ws7

python main_dino.py \
--exp_name dinov1_bs128_no_norm_last_layer \
--output_dir /dinov1_models \
--data_path /jvrielink/AIML_rot_corrected \
--arch resnet50 \
--optimizer sgd \
--lr 0.03 \
--weight_decay 1e-4 \
--weight_decay_end 1e-4 \
--global_crops_scale 0.14 1 \
--local_crops_scale 0.05 0.14 \
--batch_size_per_gpu 128 \
--num_workers 14 \
--wandb_log_freq 100 \
--saveckp_freq 20 \
--epochs 61 \
--norm_last_layer False


python main_dino.py \
--exp_name dinov1_bs128_0.9995momentum_teacher \
--output_dir /dinov1_models \
--data_path /jvrielink/AIML_rot_corrected \
--arch resnet50 \
--optimizer sgd \
--lr 0.03 \
--weight_decay 1e-4 \
--weight_decay_end 1e-4 \
--global_crops_scale 0.14 1 \
--local_crops_scale 0.05 0.14 \
--batch_size_per_gpu 128 \
--num_workers 14 \
--wandb_log_freq 100 \
--saveckp_freq 20 \
--epochs 61 \
--momentum_teacher 0.9995

python main_dino.py \
--exp_name dinov1_bs128_increased_crop_scale \
--output_dir /dinov1_models \
--data_path /jvrielink/AIML_rot_corrected \
--arch resnet50 \
--optimizer sgd \
--lr 0.03 \
--weight_decay 1e-4 \
--weight_decay_end 1e-4 \
--global_crops_scale 0.5 1 \
--local_crops_scale 0.25 0.5 \
--batch_size_per_gpu 128 \
--num_workers 14 \
--wandb_log_freq 100 \
--saveckp_freq 20 \
--epochs 61