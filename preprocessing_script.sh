#!/bin/bash

SOURCEDIR=./
cd $SOURCEDIR

data_dir=256160621
mode='train'

################################################
python preprocessing/ATL03_h5_to_csv.py --data_dir ${data_dir} --removeLand --removeIrrelevant --utm
python preprocessing/split_data_bulk.py --input_dir ${data_dir} --mode ${mode}