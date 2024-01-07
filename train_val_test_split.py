'''
Created by Yiwen Lin
Date: Jul 2023
'''
import os, json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# setting
parser = argparse.ArgumentParser(description='Train-val-test data split')
parser.add_argument('--data_dir', type=str, required=True, help='Input directory')


def split_data(file_list):
    train_list, tmp_list = train_test_split(file_list, test_size=0.2, random_state=42, shuffle=True)
    val_list, test_list = train_test_split(tmp_list, test_size=0.5, random_state=42, shuffle=True)
    print(len(train_list))
    print(len(val_list))
    print(len(test_list))

    return train_list, val_list, test_list


def create_json_file(train_list, val_list, test_list, out_dir='train_test_split'):

    os.makedirs(out_dir, exist_ok=True)

    train_json = json.dumps(train_list)
    with open(os.path.join(out_dir, 'train_file_list.json'), 'w') as f:
        f.write(train_json)

    val_json = json.dumps(val_list)
    with open(os.path.join(out_dir, 'val_file_list.json'), 'w') as f:
        f.write(val_json)

    test_json = json.dumps(test_list)
    with open(os.path.join(out_dir, 'test_file_list.json'), 'w') as f:
        f.write(test_json)


def main(args):
    # os.chdir(args.data_dir)

    undersample_test = False

    data_dir = os.path.join(args.data_dir, 'input_data')
    file_list_sf = []
    file_list_non = []
    for file in os.listdir(data_dir):
        if 'seafloor' in file:
            file_list_sf.append(file)
        else:
            file_list_non.append(file)

    # Split the two types of files into train, val and test sets in a stratified manner
    train_list_sf, val_list_sf, test_list_sf = split_data(file_list_sf)
    train_list_non, val_list_non, test_list_non = split_data(file_list_non)
    print(len(train_list_sf))

    # Combine the val and test sets for both classes
    val_all = val_list_sf + val_list_non
    test_all = test_list_sf + test_list_non
    print(len(val_all))
    print(len(test_all))

    if not undersample_test:
        ratio = 0.4
        n_samples = int(ratio * len(train_list_non))
        undersampled_files = resample(train_list_non, n_samples=n_samples, replace=False, random_state=42)
        train_all = undersampled_files + train_list_sf
        print(len(train_all))
        create_json_file(train_all, val_all, test_all)

    else:
        undersampling_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]

        for ratio in undersampling_ratios:
            # Undersample non-seafloor only files based on the current ratio
            n_samples = int(ratio * len(train_list_non))
            undersampled_files = resample(train_list_non, n_samples=n_samples, replace=False, random_state=42)

            # Concatenate the undersampled non-seafloor only files with the seafloor files
            train_all = undersampled_files + train_list_sf
            print("%1f: %d %d" % (ratio, len(undersampled_files), len(train_all)))

            out_dir = 'train_test_split_' + str(ratio)
            create_json_file(train_all, val_all, test_all, out_dir=out_dir)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)