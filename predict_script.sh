#!/bin/bash   

SOURCEDIR=./
cd $SOURCEDIR

# parameter
test_file=predict.py
npoint=8192
batch_size=1
data_root=data
threshold=0.5
num_votes=10

################################################
## predict
python ${test_file} --batch_size ${batch_size} --num_point ${npoint} \
       --data_root ${data_root} --conf --threshold ${threshold} --num_votes ${num_votes}

