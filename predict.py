"""
Modified by Yiwen Lin
Date: Jul 2023
"""
import argparse
import os
from pathlib import Path
import torch
import datetime
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Seafloor': [0, 1]}

seg_label_to_cat = {}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    return new_y.to(y.device)


def pc_denormalize(pc, pc_min, pc_max):
    for i in range(pc.shape[1]):
        pc[:, i] = (pc[:, i] + 1) / 2 * (pc_max[i] - pc_min[i]) + pc_min[i]
    return pc


def pc_normalize(pc):
    pc_min = np.empty(3, dtype=np.float64)
    pc_max = np.empty(3, dtype=np.float64)
    for i in range(pc.shape[1]):
        pc_min[i] = min(pc[:, i])
        pc_max[i] = max(pc[:, i])
        pc[:, i] = 2 * ((pc[:, i] - pc_min[i]) / (pc_max[i] - pc_min[i])) - 1
    return pc, pc_min, pc_max


class PartNormalDataset(Dataset):
    def __init__(self, root = './data', npoints=8192, class_choice=None, conf_channel=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.conf_channel = conf_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            for fn in fns:
                if os.path.splitext(os.path.basename(fn))[1] == '.txt':
                    self.meta[item].append(os.path.join(dir_point, fn))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        self.seg_classes = {'Seafloor': [0,1]}

        self.cache = {}
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            file_name = os.path.basename(fn[1])
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float64)
            if not self.conf_channel:
                point_set = data[:, [0, 1, 4]]  # use x,y,elev
            else:
                point_set = data[:, [0, 1, 4, 5]]  # use x,y,elev,signal_conf
                point_set_coor = data[:, [2, 3]] # store coordinates
                point_set[:, -1] = point_set[:, -1].astype(np.int32)

            length = len(point_set)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        point_set_normalized = point_set
        point_set_normalized[:, 0:3], pc_min, pc_max = pc_normalize(point_set[:, 0:3])

        point_set_normalized_mask = np.full(self.npoints, True, dtype=bool)
        # resample
        if length > self.npoints:
            choice = np.random.choice(length, self.npoints, replace=False)
            point_set_normalized = point_set_normalized[choice, :]
            point_set_coor = point_set_coor[choice]
        elif length < self.npoints:
            if not self.conf_channel:
                pad_point = np.ones((self.npoints-length, 3), dtype=np.float32)
            else:
                pad_point = np.ones((self.npoints - length, 3), dtype=np.float32)
                pad_conf = np.ones((self.npoints - length, 1), dtype=np.int32)
                pad_point = np.concatenate((pad_point, pad_conf), axis=1)

            point_set_normalized = np.concatenate((point_set_normalized, pad_point), axis=0)

            # create mask for point set - mask out the padded points
            pad_point_bool = np.full(self.npoints - length, False, dtype=bool)
            point_set_normalized_bool = np.full(length, True, dtype=bool)
            point_set_normalized_mask = np.concatenate((point_set_normalized_bool, pad_point_bool))

        return point_set_normalized, cls, \
                   file_name, point_set_normalized_mask, pc_min, pc_max, point_set_coor

    def __len__(self):
        return len(self.datapath)


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--conf', action='store_true', default=False, help='use confidence level')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    parser.add_argument('--data_root', type=str, required=True, help='data root file')
    parser.add_argument('--output', action='store_false', help='output test results')
    parser.add_argument('--threshold', type=float, default=0.5, help='probability threshold')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''CREATE DIR'''
    args = parse_args()
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    log_dir = Path('./log')
    log_dir.mkdir(exist_ok=True)
    log_dir = log_dir.joinpath(timestr)
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    if args.output:
        # create output folder for test output files
        output_dir = log_dir.joinpath('output_' + str(args.threshold))

        if not os.path.exists(output_dir):
            output_dir.mkdir()

    root = args.data_root

    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, conf_channel=args.conf)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=3)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 1
    num_part = 2

    '''MODEL LOADING'''
    model_name = 'pointnet2_part_seg_msg'
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, conf_channel=args.conf).to(device)
    trained_model = torch.load('./trained_model/model.pth', map_location=torch.device(device))
    # if device == 'cpu':
    #     trained_model = torch.load('./trained_model/model.pth', map_location=torch.device('cpu'))
    # else:
    #     trained_model = torch.load('./trained_model/model.pth')

    model_state_dict = {k.replace('module.', ''): v for k, v in trained_model['model_state_dict'].items()}
    classifier.load_state_dict(model_state_dict)

    thres = args.threshold

    with torch.no_grad():

        classifier = classifier.eval()
        for batch_id, (points, label, file_name, point_set_normalized_mask, pc_min, pc_max, point_set_coor) in \
                tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):

            cur_batch_size, NUM_POINT, _ = points.size()
            points, label = points.float().to(device), label.long().to(device)
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(cur_batch_size, NUM_POINT, num_part).to(device)

            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes

            cur_pred = seg_pred.cpu().numpy()
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            cur_pred_prob = np.zeros((cur_batch_size, NUM_POINT)).astype(np.float64)
            point_set_normalized_mask = point_set_normalized_mask.numpy()
            cur_pred_val_mask = []
            cur_pred_prob_mask = []

            for i in range(cur_batch_size):
                prob = np.exp(cur_pred[i, :, :])
                cur_pred_prob[i, :] = prob[:, 1]  # the probability of belonging to seafloor class
                cur_pred_val[i, :] = np.where(prob[:, 1] < thres, 0, 1)
                cur_mask = point_set_normalized_mask[i, :]
                cur_pred_prob_mask.append(cur_pred_prob[i, cur_mask])
                cur_pred_val_mask.append(cur_pred_val[i, cur_mask])

            if args.output:
                # reshape points and put it back to cpu
                points = points.transpose(2, 1)
                points = points.cpu().numpy()

                pc_min = pc_min.numpy()
                pc_max = pc_max.numpy()
                point_set_coor = point_set_coor.numpy()

                for i in range(cur_batch_size):
                    # mask out padded points
                    cur_points = points[i, :, :]
                    cur_mask = point_set_normalized_mask[i, :]
                    cur_points = cur_points[cur_mask, :]
                    # create a new point cloud array
                    output_points = np.zeros((cur_points.shape[0], 7)).astype(np.float64)
                    output_points[:, 0:3] = cur_points[:, 0:3]
                    # recover the point coordinates
                    cur_pc_min = pc_min[i, :]
                    cur_pc_max = pc_max[i, :]
                    cur_coor = point_set_coor[i, :, :]
                    output_points[:, 0:3] = pc_denormalize(output_points[:, 0:3], cur_pc_min, cur_pc_max)
                    # output coordinates
                    output_points[:, 3:5] = cur_coor
                    # output class and probability
                    output_points[:, 5] = cur_pred_val_mask[i]
                    output_points[:, 6] = cur_pred_prob_mask[i]
                    output_file = file_name[i]
                    output_path = os.path.join(output_dir, output_file)
                    np.savetxt(output_path, output_points, delimiter=' ', fmt='%.4f')

    # Combine all the sub-files to the original beam files
    post_process_script = 'post_process.py'
    data_dir = 'output_' + str(args.threshold)
    out_dir = data_dir + '_merge'
    post_process_command = 'python ' + post_process_script + ' --log_dir ' + str(log_dir) + ' --data_dir ' \
                           + data_dir + ' --output_dir ' + out_dir

    return_code = os.system(post_process_command)
    if return_code != 0:
        print("Run post process script error")


if __name__ == '__main__':
    args = parse_args()
    main(args)
