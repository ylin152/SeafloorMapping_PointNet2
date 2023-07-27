'''
Created by Yiwen Lin
Date: Jul 2023
'''
import os, argparse
import pandas as pd


def refraction_correction_approx(b_z, w_z):
    b_z = b_z + 0.25416 * (w_z - b_z)
    return b_z


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, help='experiment root')
    parser.add_argument('--data_dir', type=str, help='input data directory')
    parser.add_argument('--file_list', type=str, default='file_list.txt', help='a list of original files in txt format')
    parser.add_argument('--output_dir', type=str, help='output directory')

    return parser.parse_args()


def main(args):
    log_dir = args.log_dir
    file_dir = os.path.join(log_dir, args.data_dir)
    output_dir = os.path.join(log_dir, args.output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    col = ['x', 'y', 'elev', 'lon', 'lat', 'label', 'prob']

    file_list = []
    for sub_file in os.listdir(file_dir):
        if 'seafloor' in sub_file:
            basename = os.path.splitext(sub_file)[0]
            basename = basename.strip('_seafloor')
            ext = os.path.splitext(sub_file)[1]
            os.rename(os.path.join(file_dir, sub_file), os.path.join(file_dir, basename+ext))
            sub_file = basename+ext
        file_list.append(os.path.splitext(sub_file)[0][:-3])

    file_list = set(file_list)

    for file in file_list:
        sub_file_list = []
        for sub_file in os.listdir(file_dir):
            if file in sub_file:
                df_sub_file = pd.read_csv(os.path.join(file_dir, sub_file), sep=' ', names=col)
                sub_file_list.extend(df_sub_file.to_numpy().tolist())
        df = pd.DataFrame(sub_file_list, columns=col)
        # convert label column to integer
        df['label'] = df['label'].astype(int)
        output_file = os.path.join(output_dir, file + '.txt')  # or output to csv file
        df.to_csv(output_file, sep=' ', index=False, header=False)


if __name__ == '__main__':
    args = parse_args()
    args.log_dir = './log/2023-07-26_19-32-32'
    args.data_dir = 'output_0.5'
    args.output_dir = 'output_0.5_merge'
    main(args)
