
python -m torch.distributed.launch --nproc_per_node=1 main_dino.py \
--arch resnet50 \
--optimizer sgd \
--lr 0.03 \
--weight_decay 1e-4 \
--weight_decay_end 1e-4 \
--global_crops_scale 0.14 1 \
--local_crops_scale 0.05 0.14 \
--data_path /mnt/sdb1/Data_remote/AIML_rot_corrected \
--output_dir debug_output \
--batch_size_per_gpu 4