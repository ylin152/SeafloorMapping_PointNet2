#!/bin/bash   

SOURCEDIR=./
cd $SOURCEDIR

# parameter
split_file=train_val_test_split.py
train_file=train_partseg.py
model=pointnet2_part_seg_msg
epoch_num=550
npoint=8192
batch_size=32
learning_rate=0.0001
lr_decay=1
step_size=1
loss_weight=3.0
data_root=data_8192_2
ckpt=

################################################
## split training data
python ${split_file} --data_dir ${data_root}

## training
python ${train_file} --model ${model} --epoch ${epoch_num} --npoint ${npoint} \
       --batch_size ${batch_size} --learning_rate ${learning_rate}\
       --data_root ${data_root} --conf --loss_weight ${loss_weight}\
       --lr_decay ${lr_decay} --ckpt ${ckpt} --step_size ${step_size}
