#!/usr/bin/env bash


# ./run_pretrain_no_augmentation.sh

# pause 30s

# ./run_pretrain_no_multi_crop.sh

# pause 30s

./run_pretrain_no_local_to_global.sh

pause 30s

./run_pretrain_no_local_to_global_no_multicrop.sh

pause 30s

./run_pretrain_no_augmentation_smoothing.sh


