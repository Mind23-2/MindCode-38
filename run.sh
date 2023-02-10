#! /bin/bash

#ps -aux | grep "python -u ../train_CityScapes.py" | awk '{print $2}' | xargs kill -9

#mkdir ./log_single_device
cd ./log_single_device

encode="false"
export RANK_SIZE=1
export DEVICE_ID=4

python -u ../train_CityScapes.py \
    --lr 5e-4 \
    --repeat 1 \
    --run_distribute false \
    --run_online false \
    --encode ${encode} \
    --epoch 150 \
    --run_eval true \
    --save_path $(pwd) \
    --pretrain /home/gpf/ERFNet/log_single_device/ERFNet_1-11_496.ckpt \
    --start_epoch 65 \
    --attach_decoder false \
    > log.txt 2>&1

python -u ../eval.py \
  --data_path /home/gpf/cityscapes \
  --run_distribute false \
  --encode ${encode} \
  --model_root_path $(pwd) \
  --device_id ${DEVICE_ID} \
  > log_eval.txt 2>&1 &