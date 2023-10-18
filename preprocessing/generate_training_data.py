# Yiwen Lin, September 2022
# Pass annotation to original beam files and split files by equal number

import os, math, glob
import pandas as pd
import numpy as np
import argparse


def find_seafloor(file_list):
    # create a dictionary for storing the seafloor locations of each beam; default to -1
    seafloor_dict = {}
    tracks = ['1l', '1r', '2l', '2r', '3l', '3r']
    for track in tracks:
        seafloor_dict[track] = []

    for file in file_list:
        try:
            df = pd.read_csv(file, sep=',', header=None)  # default sep is ','
            if not df.empty:
                # get seafloor point locations
                seafloor_loc = df.iloc[:, [0, -1]].to_numpy().tolist()
                # store the seafloor location by beam file
                fbasename = os.path.basename(file)
                for track in tracks:
                    if track in fbasename:
                        seafloor_dict[track].extend(seafloor_loc)
                        break
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            pass

    return seafloor_dict


def annotate_seafloor(df, seafloor_loc_h):
    if len(seafloor_loc_h) > 0:
        # method 1
        df_loc = df.loc[:, ["y", "elev"]].to_numpy().tolist()
        for i, item in enumerate(df_loc):
            if item in seafloor_loc_h:
                df.loc[i, "annotation"] = 1
            # elif item in seafloor_loc_l:
            #     df.loc[i, "annotation"] = 2

    return df


def split_by_npoints(file_list, output_dir, mode='train', npoints=8192, overwrite=True):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for file in file_list:
        df = pd.read_csv(file)
        df["signal_conf_ph"] = df["signal_conf_ph"] - 2
        file_base = os.path.basename(file)

        # get number of row
        nrow = df.shape[0]
        # get number of sub-files
        nsubregion = math.ceil(nrow / npoints)
        # store the start and end index of each sub-file into a list
        subregion_index = []
        if nsubregion > 1:
            for j in range(nsubregion - 1):
                start_index = j * npoints
                end_index = (j + 1) * npoints
                index = (start_index, end_index)
                subregion_index.append(index)
        # for last sub-file or for files that have less than 40000 points; discard the files of only 1 point
        j = nsubregion - 1
        start_index = j * npoints
        end_index = nrow
        if end_index - start_index > 1:
            index = (start_index, end_index)
            subregion_index.append(index)

        for i in range(len(subregion_index)):
            if mode == 'test':
                df_subregion = df.iloc[subregion_index[i][0]:subregion_index[i][1],
                    df.columns.get_indexer(["x", "y", "elev", "lon", "lat", "class", "signal_conf_ph"])]
                output_filename = os.path.splitext(file_base)[0] + '_' + str(i + 1).zfill(2) + '.txt'
                output_file_path = os.path.join(output_dir, output_filename)
                # output file to pointnet++ input data format
                if output_file_path and not overwrite:
                    continue
                df_subregion.to_csv(output_file_path, header=None, index=None, sep=' ')
            elif mode == 'train':
                df["annotation"] = 0
                df_subregion = df.iloc[subregion_index[i][0]:subregion_index[i][1],
                    df.columns.get_indexer(["x", "y", "elev", "lon", "lat", "class", "signal_conf_ph",
                                            "annotation"])]
                output_filename = os.path.splitext(file_base)[0] + '_' + str(i + 1).zfill(2) + '.txt'
                output_file_path = os.path.join(output_dir, output_filename)
                # output file to pointnet++ input data format
                if output_file_path and not overwrite:
                    continue
                df_subregion.to_csv(output_file_path, header=None, index=None, sep=' ')
            else:
                df["annotation"] = 0
                df_subregion = df.iloc[subregion_index[i][0]:subregion_index[i][1],
                    df.columns.get_indexer(["x", "y", "elev", "lon", "lat", "class", "signal_conf_ph",
                                            "annotation"])]
                output_filename = os.path.splitext(file_base)[0] + '_' + str(i + 1).zfill(2) + '.csv'
                output_file_path = os.path.join(output_dir, output_filename)
                if output_file_path and not overwrite:
                    continue
                df_subregion.to_csv(output_file_path, index=None, sep=',')

    return subregion_index


# setting
parser = argparse.ArgumentParser(description='Generate training files')
parser.add_argument('input_dir', help='Input directory')
parser.add_argument('output_dir', help='Output directory')
parser.add_argument('--split_method', help='The split method for generating subset files, choose between npoints or latitude')
parser.add_argument('--npoints', type=int, default=8192, help='Number of points used for splitting')
parser.add_argument('--itvlat', type=int, default=1, help='Interval of latitude used for splitting')
parser.add_argument('--overwrite', action='store_true', help='Whether overwrite the existing output files or not')
parser.add_argument('--split_flag', action='store_true')


def generate_annotation(dir1, output_dir, split_method='npoints', npoints=8192, lat_interval=None, overwrite=True, split_flag=True):
    # annotation file list
    file_list_h = []
    file_list_l = []

    # original file list
    file_list = []
    file_all = glob.glob(dir1 + '*.csv')
    for file in file_all:
        # find original beam files based on file name and extension
        fname = os.path.splitext(os.path.basename(file))[0]
        ext = os.path.splitext(os.path.basename(file))[1]
        if ext == '.csv' and 'annotated' not in fname:
            file_list.append(file)
    for file in file_list:
        filename = os.path.splitext(os.path.basename(file))[0]
        # find annotation files
        file_h = glob.glob(dir1 + filename + '_annotated_h*')
        file_l = glob.glob(dir1 + filename + '_annotated_l*')
        # if any of the corresponding annotation file doesn't exist, show warning message
        # if not file_h:
        #     print("Warning, missing high probability annotation file for " + filename)
        # if not file_l:
        #     print("Warning, missing low probability annotation file for " + filename)
        # extend the annotation files to the corresponding lists
        file_list_h.extend(file_h)
        file_list_l.extend(file_l)

    # find and store seafloor point locations
    # print("Finding seafloor point locations from annotation files...")
    # for high probability annotation
    seafloor_dict_h = find_seafloor(file_list_h)
    # for low probability annotation
    # seafloor_dict_l = find_seafloor(file_list_l)

    # pass the seafloor annotation to the original file and split it to new sub-files of equal point number
    # print("Generating files with annotation...")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for file in file_list:
        df = pd.read_csv(file)
        file_base = os.path.basename(file)
        # pass annotation
        # at first annotate every point as '0'
        df["annotation"] = 0
        df["signal_conf_ph"] = df["signal_conf_ph"] - 2

        for track in ['1l', '1r', '2l', '2r', '3l', '3r']:
            if track in file_base:
                # find seafloor points and annotate them as '1'
                df = annotate_seafloor(df, seafloor_dict_h[track])
                break

        if split_flag is False:
            # output files with annotation
            output_filename = os.path.splitext(file_base)[0] + '_annotated.csv'
            output_file_path = os.path.join(dir1, output_filename)
            df = df[["lon", "lat", "elev", "annotation"]]
            df.to_csv(output_file_path, index=None, sep=',')

        else:
            # split files - output training files
            # split by npoints or latitude
            if split_method == 'npoints':
                subregion_index = split_by_npoints(df, npoints=npoints)
            else:
                # subregion_index = split_by_lat(df, lat_interval=lat_interval)
                print('Currently only support splitting by number of points')

            # output each sub-file
            for i in range(len(subregion_index)):
                df_subregion = df.iloc[subregion_index[i][0]:subregion_index[i][1],
                               df.columns.get_indexer(["x", "y", "lon", "lat", "elev", "signal_conf_ph", "class",
                                                       "annotation"])]

                # if contains seafloor points, add '_seafloor' to filename
                if np.any(df_subregion["annotation"].to_numpy() != 0):
                    output_filename = os.path.splitext(file_base)[0] + '_' + str(i + 1).zfill(2) + '_seafloor' + '.txt'
                else:
                    output_filename = os.path.splitext(file_base)[0] + '_' + str(i + 1).zfill(2) + '.txt'

                output_file_path = os.path.join(output_dir, output_filename)
                # output file to pointnet++ input data format
                if output_file_path and not overwrite:
                    print(output_file_path + " already exists, skip")
                    continue
                df_subregion.to_csv(output_file_path, header=None, index=None, sep=' ')


def main(args):
    # parse arguments
    dir1 = args.input_dir
    output_dir = args.output_dir
    split_method = args.split_method
    npoints = args.npoints
    lat_interval = args.itvlat
    overwrite = args.overwrite
    split_flag = args.split_flags
    generate_annotation(dir1, output_dir, split_method=split_method, npoints=npoints, lat_interval=lat_interval,
                        overwrite=overwrite, split_flag=split_flag)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
