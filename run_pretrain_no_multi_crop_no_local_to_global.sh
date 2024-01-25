# Num workers was 14 (optimal) but caused OOM errors for RAM on ws7

python main_dino.py \
--exp_name test_no_multi_crop_no_local_to_global \
--output_dir /home/jvrielink/data/dinov1_models \
--data_path /home/jvrielink/AIML_rot_corrected \
--arch resnet50 \
--optimizer sgd \
--lr 0.03 \
--weight_decay 1e-4 \
--weight_decay_end 1e-4 \
--global_crops_scale 0.14 1 \
--local_crops_scale 0.14 1 \
--batch_size_per_gpu 128 \
--num_workers 10 \
--wandb_log_freq 100 \
--saveckp_freq 10 \
--epochs 301 \
--local_crops_number 2 \
--local_crop_size 224


