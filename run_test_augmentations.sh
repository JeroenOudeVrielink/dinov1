#!/usr/bin/env bash

# Num workers was 14 (optimal) but caused OOM errors for RAM on ws7

python main_dino.py \
--exp_name dinov1_bs128_no_horizontal_flip \
--output_dir ~/data/dinov1_models \
--data_path ~/AIML_rot_corrected \
--arch resnet50 \
--optimizer sgd \
--lr 0.03 \
--weight_decay 1e-4 \
--weight_decay_end 1e-4 \
--global_crops_scale 0.14 1 \
--local_crops_scale 0.05 0.14 \
--batch_size_per_gpu 64 \
--num_workers 14 \
--wandb_log_freq 100 \
--saveckp_freq 20 \
--epochs 61 \
--p_horizontal_flip 0.0


python main_dino.py \
--exp_name dinov1_bs128_no_color_jitter \
--output_dir ~/data/dinov1_models \
--data_path ~/AIML_rot_corrected \
--arch resnet50 \
--optimizer sgd \
--lr 0.03 \
--weight_decay 1e-4 \
--weight_decay_end 1e-4 \
--global_crops_scale 0.14 1 \
--local_crops_scale 0.05 0.14 \
--batch_size_per_gpu 64 \
--num_workers 14 \
--wandb_log_freq 100 \
--saveckp_freq 20 \
--epochs 61 \
--p_color_jitter 0.0


python main_dino.py \
--exp_name dinov1_bs128_no_solarization \
--output_dir ~/data/dinov1_models \
--data_path ~/AIML_rot_corrected \
--arch resnet50 \
--optimizer sgd \
--lr 0.03 \
--weight_decay 1e-4 \
--weight_decay_end 1e-4 \
--global_crops_scale 0.14 1 \
--local_crops_scale 0.05 0.14 \
--batch_size_per_gpu 64 \
--num_workers 14 \
--wandb_log_freq 100 \
--saveckp_freq 20 \
--epochs 61 \
--p_solarization 0.0


python main_dino.py \
--exp_name dinov1_bs128_no_gaussian_blur \
--output_dir ~/data/dinov1_models \
--data_path ~/AIML_rot_corrected \
--arch resnet50 \
--optimizer sgd \
--lr 0.03 \
--weight_decay 1e-4 \
--weight_decay_end 1e-4 \
--global_crops_scale 0.14 1 \
--local_crops_scale 0.05 0.14 \
--batch_size_per_gpu 64 \
--num_workers 14 \
--wandb_log_freq 100 \
--saveckp_freq 20 \
--epochs 61 \
--disable_gaussian_blur True




