"""
Modified by Yiwen Lin
Date: Jul 2023
"""
import argparse
import os
from pathlib import Path
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix

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
    def __init__(self, root='./data', npoints=8192, conf_channel=True):
        self.npoints = npoints
        self.root = root
        self.conf_channel = conf_channel
        self.cat = {}
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        self.meta = {}
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, '111')
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

    def __getitem__(self, index):
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
            point_set_coor = data[:, [2, 3]]  # store coordinates
            point_set[:, -1] = point_set[:, -1].astype(np.int32)

        seg = data[:, -1].astype(np.int32)

        point_set_normalized = point_set
        point_set_normalized[:, 0:3], pc_min, pc_max = pc_normalize(point_set[:, 0:3])

        point_set_normalized_mask = np.full(self.npoints, True, dtype=bool)
        # resample
        if len(seg) > self.npoints:
            choice = np.random.choice(len(seg), self.npoints, replace=False)
            point_set_normalized = point_set_normalized[choice, :]
            seg = seg[choice]
            point_set_coor = point_set_coor[choice]
        elif len(seg) < self.npoints:
            if not self.conf_channel:
                pad_point = np.ones((self.npoints - len(seg), 3), dtype=np.float32)
            else:
                pad_point = np.ones((self.npoints - len(seg), 3), dtype=np.float32)
                pad_conf = np.ones((self.npoints - len(seg), 1), dtype=np.int32)
                pad_point = np.concatenate((pad_point, pad_conf), axis=1)

            point_set_normalized = np.concatenate((point_set_normalized, pad_point), axis=0)

            # create mask for point set - mask out the padded points
            pad_point_bool = np.full(self.npoints - len(seg), False, dtype=bool)
            point_set_normalized_bool = np.full(len(seg), True, dtype=bool)
            point_set_normalized_mask = np.concatenate((point_set_normalized_bool, pad_point_bool))

            pad_seg = np.zeros(self.npoints - len(seg), dtype=np.int32)
            seg = np.concatenate((seg, pad_seg), axis=0)

        return point_set_normalized, cls, seg, \
               file_name, point_set_normalized_mask, pc_min, pc_max, point_set_coor

    def __len__(self):
        return len(self.datapath)


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--ckpt', type=str, default=None, help='model checkpoint')
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
    experiment_dir = 'log/part_seg/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    if args.output:
        # create output folder for test output files
        if args.ckpt:
            output_dir = Path(experiment_dir + '/output_all_' + str(args.ckpt).split('.')[0])
            data_dir = 'output_all_' + str(args.ckpt).split('.')[0]
        else:
            output_dir = Path(experiment_dir + '/output_all')
            data_dir = 'output_all'

        if not os.path.exists(output_dir):
            output_dir.mkdir()

    root = args.data_root

    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, conf_channel=args.conf)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=3)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 1
    num_part = 2

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, conf_channel=args.conf).to(device)
    # if want to use checkpoint for testing
    if args.ckpt:
        checkpoint = torch.load(os.path.join(experiment_dir, 'checkpoints', args.ckpt))
    else:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/model.pth')

    model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    classifier.load_state_dict(model_state_dict)

    thres = args.threshold

    with torch.no_grad():

        tp_acc, fp_acc, fn_acc = 0, 0, 0

        test_metrics = {}
        part_ious = {part: [] for part in seg_classes['Seafloor']}

        classifier = classifier.eval()
        for batch_id, (points, label, target, file_name, point_set_normalized_mask, pc_min, pc_max, point_set_coor) in \
                tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):

            print(batch_id)

            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).to(device)

            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes

            cur_pred = seg_pred.cpu().numpy()
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            cur_pred_prob = np.zeros((cur_batch_size, NUM_POINT)).astype(np.float64)
            target = target.cpu().data.numpy()
            point_set_normalized_mask = point_set_normalized_mask.numpy()
            cur_pred_val_mask = []
            cur_pred_prob_mask = []
            target_mask = []

            for i in range(cur_batch_size):
                prob = np.exp(cur_pred[i, :, :])
                cur_pred_prob[i, :] = prob[:, 1]  # the probability of belonging to seafloor class
                cur_pred_val[i, :] = np.where(prob[:, 1] < thres, 0, 1)
                cur_mask = point_set_normalized_mask[i, :]
                cur_pred_prob_mask.append(cur_pred_prob[i, cur_mask])
                cur_pred_val_mask.append(cur_pred_val[i, cur_mask])
                target_mask.append(target[i, cur_mask])

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

            for i in range(cur_batch_size):
                segp = cur_pred_val_mask[i]
                segl = target_mask[i]
                for l in seg_classes['Seafloor']:
                    if np.sum(segl == l) == 0:
                        continue
                    else:
                        iou = np.sum((segl == l) & (segp == l)) / float(
                            np.sum(segl == l))
                        part_ious[l].append(iou)

            target_mask = np.hstack(target_mask)
            cur_pred_val_mask = np.hstack(cur_pred_val_mask)

            # calculate metric - F1 score
            cm = confusion_matrix(target_mask, cur_pred_val_mask)  # sklearn
            if cm.shape[0] == 1:
                tp, fp, fn = 0, 0, 0
            else:
                # since we don't care about non-seafloor class, index start from 1
                tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]

            # accumulate tp, fp, fn
            tp_acc += tp
            fp_acc += fp
            fn_acc += fn

        # calculate on the entire test dataset
        precision = tp_acc / (tp_acc + fp_acc) if (tp_acc + fp_acc) > 0 else 1.0
        recall = tp_acc / (tp_acc + fn_acc) if (tp_acc + fn_acc) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        test_metrics['Precision'] = precision
        test_metrics['Recall'] = recall
        test_metrics['F1 score'] = f1
        log_string('Precision: %.5f' % test_metrics['Precision'])
        log_string('Recall: %.5f' % test_metrics['Recall'])
        log_string('F1 score: %.5f' % test_metrics['F1 score'])

        mean_part_iou = []
        for part in sorted(part_ious.keys()):
            part_ious[part] = np.mean(part_ious[part])
            log_string('eval IoU of part %d: %f' % (part, part_ious[part]))
            mean_part_iou.append(part_ious[part])
        mean_part_iou = np.mean(mean_part_iou)
        test_metrics['part_avg_iou'] = mean_part_iou

    log_string('Part avg mIOU is: %.5f' % test_metrics['part_avg_iou'])

    # Combine all the sub-files together
    post_process_script = 'post_process.py'
    out_dir = data_dir + '_merge'
    post_process_command = 'python ' + post_process_script + ' --log_dir ' + experiment_dir \
                           + ' --data_dir ' + data_dir + ' --output_dir ' + out_dir

    return_code = os.system(post_process_command)
    if return_code != 0:
        print("Run post process script error")


if __name__ == '__main__':
    args = parse_args()
    main(args)
