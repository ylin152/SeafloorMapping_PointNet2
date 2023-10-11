#!/bin/bash

SOURCEDIR=./
cd $SOURCEDIR

data_root=256160621

################################################
python pre-processing/ATL03_h5_to_csv.py --data_dir ${data_root} --removeLand --removeIrrelevant --utm
python pre-processing/split_data_bulk.py --input_dir ${data_root}