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
## pre-processing
python pre-processing/ATL03_h5_to_csv.py --data_dir ${data_root} --removeLand --removeIrrelevant --utm
python pre-processing/split_data_bulk.py --input_dir ${data_root}

## predict
python predict.py --batch_size ${batch_size} --num_point ${npoint} \
       --data_root ${data_root} --conf --threshold ${threshold} --num_votes ${num_votes}