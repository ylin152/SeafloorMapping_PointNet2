# Yiwen Lin, July 2022
# Pre-process files

import glob, os
from generate_training_data import generate_annotation
import argparse

# setting
parser = argparse.ArgumentParser(description='Convert ATL03 to CSV file')
parser.add_argument('--input_dir', type=str, required=True, help='Input directory')

def main(args):
    input_dir = args.input_dir
    output_dir = os.path.join(input_dir, '1')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(os.path.join(input_dir, 'category.txt'), 'w') as f_obj:
        f_obj.write('Seafloor 1')
    sub_dirs = glob.glob(input_dir+'/*/')
    for sub_dir in sub_dirs:
        generate_annotation(sub_dir, output_dir, split_method='npoints', lat_interval=1, npoints=8192, overwrite=True)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)