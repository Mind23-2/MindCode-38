#! /bin/bash

ps -aux | grep "python -u ../../train_CityScapes.py" | awk '{print $2}' | xargs kill -9

encode="false"
export HCCL_CONNECT_TIMEOUT=600
export RANK_SIZE=4
export RANK_TABLE_FILE=/home/gpf/rank_table_4pcs.json
devices=(4 5 6 7)

mkdir ./log
cd ./log

# train
for((i=0;i<RANK_SIZE;i++))
do
{
#  mkdir ./log$i
  cd ./log$i
  export RANK_ID=$i
  export DEVICE_ID=${devices[i]}
  echo "start training for rank $i, device $DEVICE_ID"

  python -u ../../train_CityScapes.py \
    --lr 8e-4 \
    --repeat 2 \
    --run_distribute true \
    --run_online false \
    --encode ${encode} \
    --epoch 150 \
    --run_eval true \
    --save_path $(pwd)/ \
    --pretrain /home/gpf/ERFNet/log/log0/ERFNet-65_248.ckpt \
    --attach_decoder false \
    --start_epoch 65 \
    > log.txt 2>&1
  cd ../
} &
done
wait

# eval
cd ./log0
for((i=0;i<RANK_SIZE;i++))
do
{
  export RANK_ID=$i
  export DEVICE_ID=${devices[i]}
  echo "start eval for rank $i, device $DEVICE_ID"
  python -u ../../eval.py \
    --data_path /home/gpf/cityscapes \
    --run_distribute true \
    --encode ${encode} \
    --model_root_path $(pwd) \
    --device_id ${devices[i]} \
    > log${i}_eval.txt 2>&1 &
}
done
