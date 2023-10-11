#!/bin/bash   

SOURCEDIR=./
cd $SOURCEDIR

# parameter
data_root=256160621
npoint=8192
batch_size=1
threshold=0.5
num_votes=10

################################################
## predict
python predict.py --batch_size ${batch_size} --num_point ${npoint} \
       --data_root ${data_root} --conf --threshold ${threshold} --num_votes ${num_votes}