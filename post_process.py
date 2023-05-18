import os, argparse
import pandas as pd


def refraction_correction_approx(b_z, w_z):
    b_z = b_z + 0.25416 * (w_z - b_z)
    return b_z


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='input data directory')
    parser.add_argument('--file_list', type=str, help='a list of original files in txt format')
    parser.add_argument('--output_dir', type=str, help='output directory')

    return parser.parse_args()


def main():
    # dir = args.data_dir
    # output_dir = args.output_dir
    # file_list_dir = args.file_list
    file_dir = 'data17/111'
    output_dir = 'data17_merge3'
    file_list_dir = 'file_list.txt'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(file_list_dir, 'r') as f_obj:
        file_list = [file.rstrip('\n') for file in f_obj.readlines()]

    # file_dict = dict(file_list)

    col = ['x', 'y', 'lon', 'lat', 'elev', 'label', 'prob']
    col = ['x', 'y', 'lon', 'lat', 'elev', 'signal_conf', 'label']

    for file in file_list:
        for track in ['1l', '1r', '2l', '2r', '3l', '3r']:
            sub_file_list = []
            for sub_file in os.listdir(file_dir):
                if file in sub_file and track in sub_file:
                    sub_file = os.path.join(file_dir, sub_file)
                    df_sub_file = pd.read_csv(sub_file, sep=' ', names=col)

                    # refraction correction
                    b_elev = df_sub_file.loc[df_sub_file['label'] == 1, ['elev']].to_numpy().tolist()
                    b_coor = df_sub_file.loc[df_sub_file['label'] == 1, ['x', 'y']].to_numpy().tolist()
                    # if flat water surface
                    # w_elev = df_sub_file['elev'].max()
                    # if not flat
                    w_elev = df_sub_file.loc[df_sub_file['x', 'y'] == b_coor, ['elev']].to_numpy().tolist()
                    b_elev = refraction_correction_approx(b_elev, w_elev)
                    df_sub_file.loc[df_sub_file['label'] == 1, ['elev']] = b_elev

                    sub_file_list.extend(df_sub_file.to_numpy().tolist())
            df = pd.DataFrame(sub_file_list, columns=col)
            output_file = os.path.join(output_dir, file + '_' + track + '.txt')  # or output to csv file
            df.to_csv(output_file, sep=' ', index=False)




if __name__ == '__main__':
    # args = parse_args()
    main()
