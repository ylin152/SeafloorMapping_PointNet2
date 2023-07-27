#!/bin/bash   

SOURCEDIR=./
cd $SOURCEDIR

# parameter
test_file=test_partseg.py
npoint=8192
batch_size=1
data_root=data
log_dir=2023-06-21_07-58-14
## if use checkpoints
# ckpt=ckpt_550.pth
threshold=0.5
num_votes=10

################################################
## test
python ${test_file} --batch_size ${batch_size} --log_dir ${log_dir}\
       --num_point ${npoint} --data_root ${data_root} --conf\
       --ckpt ${ckpt} --threshold ${threshold} --num_votes ${num_votes}

