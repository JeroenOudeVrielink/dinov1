# docker run --gpus all -v $pwd -w /code -it pytorch-test
docker run \
-it \
--rm \
-v $(pwd):/code \
--gpus '"device=1"' \
--mount type=bind,src=/home/jvrielink,target=/jvrielink \
--mount type=bind,src=/home/jvrielink/data/dinov1_models,target=/dinov1_models \
--shm-size 64G \
jvrielink/pytorch_dinov1