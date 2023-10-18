# Yiwen Lin, July 2022
# Split files into sub-files

import glob, os, sys
import importlib
import argparse

basedir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(basedir)

# setting
parser = argparse.ArgumentParser(description='Convert ATL03 to CSV file')
parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
parser.add_argument('--mode', type=str, default='train', help='Data splitting mode')


def split(input_dir, mode='train'):
    print("Start splitting data...")

    if mode == 'train' or mode == 'manual':
        output_dir = os.path.join(input_dir, 'split_data')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
    elif mode == 'test':
        output_dir = os.path.join(input_dir, 'input_data')
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    # with open(os.path.join(input_dir, 'category.txt'), 'w') as f_obj:
    #     f_obj.write('Seafloor 1')

    generate_training_data = importlib.import_module('generate_training_data')

    sub_dirs = glob.glob(input_dir+'/csv_data/')
    for sub_dir in sub_dirs:
        # original file list
        file_list = []
        file_all = glob.glob(sub_dir + '/*')
        for file in file_all:
            # find original beam files based on file name and extension
            fname = os.path.splitext(os.path.basename(file))[0]
            ext = os.path.splitext(os.path.basename(file))[1]
            if fname.endswith('N') or fname.endswith('S'):
                file_list.append(file)
            generate_training_data.split_by_npoints(file_list, output_dir, mode)

        # generate_training_data.generate_annotation(sub_dir, output_dir, split_method='npoints', lat_interval=1, npoints=8192, overwrite=True)


def main(args):
    input_dir = args.input_dir
    split(input_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)