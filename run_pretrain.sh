
python main_dino.py \
--exp_name dino_debug \
--output_dir ~/data/dinov1_models \
--data_path ~/AIML_rot_corrected \
--arch resnet50 \
--optimizer sgd \
--lr 0.03 \
--weight_decay 1e-4 \
--weight_decay_end 1e-4 \
--global_crops_scale 0.14 1 \
--local_crops_scale 0.05 0.14 \
--batch_size_per_gpu 128 \
--num_workers 4 \
--wandb_log_freq 100 \
--epochs 10

