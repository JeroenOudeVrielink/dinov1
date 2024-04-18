#!/usr/bin/env bash

# Num workers was 14 (optimal) but caused OOM errors for RAM on ws7

cd ../..

python main_dino.py \
--exp_name dinov1_b128_out_dim8192_ep300 \
--output_dir /dinov1_models \
--data_path /AIML_rot_corrected \
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
--saveckp_freq 10 \
--epochs 301 \
--out_dim 8192
