#! /bin/bash

#export MS_RDR_ENABLE=1
#export GLOG_v=1

ps -aux | grep "python -u ../../train_CityScapes.py" | awk '{print $2}' | xargs kill -9

encode="true"
export HCCL_CONNECT_TIMEOUT=600
export RANK_SIZE=8
export RANK_TABLE_FILE=/root/gpf/rank_table_8pcs.json
devices=(0 1 2 3 4 5 6 7)

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
    --lr 1e-4 \
    --repeat 4 \
    --run_distribute true \
    --run_online false \
    --encode ${encode} \
    --epoch 150 \
    --run_eval true \
    --save_path $(pwd)/ \
    --pretrain '' \
    --start_epoch 0 \
    --attach_decoder false \
    --early_stop_epoch 9999\
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
