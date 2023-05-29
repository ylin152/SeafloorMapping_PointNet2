"""
Modified by Yiwen
05/2023
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
from torchmetrics.functional import average_precision
# from torchmetrics.functional import confusion_matrix
from sklearn.metrics import confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'Seafloor': [0, 1]}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


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
    def __init__(self, root = './data', npoints=8192, normal_channel=True):
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel
        self.cat = {}
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')

        dir_point = os.path.join(self.root, '111')

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
                    # token = (os.path.splitext(os.path.basename(fn))[0])
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
        print(fn[1])
        print(file_name)
        cat = self.datapath[index][0]
        cls = self.classes[cat]
        cls = np.array([cls]).astype(np.int32)
        data = np.loadtxt(fn[1]).astype(np.float64)
        if not self.normal_channel:
            point_set = data[:, [0, 1, 4]]  # use x,y,elev
        else:
            point_set = data[:, [0, 1, 4, 5]]  # use x,y,elev,signal_conf
            point_set[:, -1] = point_set[:, -1].astype(np.int32)

        seg = data[:, -1].astype(np.int32)
        # for only one class
        seg[seg == 2] = 0

        point_set_normalized = point_set
        point_set_normalized[:, 0:3], pc_min, pc_max = pc_normalize(point_set[:, 0:3])

        point_set_normalized_mask = np.full(self.npoints, True, dtype=bool)
        # resample
        if len(seg) > self.npoints:
            choice = np.random.choice(len(seg), self.npoints, replace=False) #replace=True
            point_set_normalized = point_set_normalized[choice, :]
            seg = seg[choice]
        elif len(seg) < self.npoints:
            if not self.normal_channel:
                pad_point = np.ones((self.npoints-len(seg), 3), dtype=np.float32)
            else:
                pad_point = np.ones((self.npoints - len(seg), 3), dtype=np.float32)
                pad_conf = np.ones((self.npoints - len(seg), 1), dtype=np.int32)
                pad_point = np.concatenate((pad_point, pad_conf), axis=1)

            point_set_normalized = np.concatenate((point_set_normalized, pad_point), axis=0)

            # create mask for point set - mask out the padded points
            pad_point_bool = np.full(self.npoints - len(seg), False, dtype=bool)
            # pad_point_bool = np.zeros(self.npoints - len(seg), dtype=bool)
            point_set_normalized_bool = np.full(len(seg), True, dtype=bool)
            # point_set_normalized_bool = np.ones(len(seg), dtype=bool)
            point_set_normalized_mask = np.concatenate((point_set_normalized_bool, pad_point_bool))

            pad_seg = np.zeros(self.npoints-len(seg), dtype=np.int32)
            seg = np.concatenate((seg, pad_seg), axis=0)

            # choice = np.random.choice(len(seg), self.npoints, re
            # place=True)
            # # resample
            # point_set = point_set[choice, :]
            # seg = seg[choice]

        return point_set_normalized, cls, seg, file_name, point_set_normalized_mask, pc_min, pc_max

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
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
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
        else:
            output_dir = Path(experiment_dir + '/output_all')

        # exist_ok = True doesn't create folder if it already exists and doesn't raise an error
        if not os.path.exists(output_dir):
            output_dir.mkdir()

        # if os.path.exists(output_dir) is False:
        #     output_dir.mkdir()
        # else:  # if output folder already exists
        #     counter = 1
        #     while os.path.exists(output_dir):
        #         dir_name = '/output' + "_" + str(counter)
        #         output_dir = Path(experiment_dir + dir_name)
        #         counter += 1
        #     output_dir.mkdir()

    # root = 'data/test5/'
    root = args.data_root

    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=3)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 1
    num_part = 2

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    # if want to use checkpoint for testing
    if args.ckpt:
        checkpoint = torch.load(os.path.join(experiment_dir, 'checkpoints', args.ckpt))
    else:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    thres = args.threshold

    with torch.no_grad():

        tp_acc, fp_acc, fn_acc = 0, 0, 0

        test_metrics = {}
        f1_acc = []
        seg_pred_all = []
        target_all = []

        # shape_ious = {cat: [] for cat in seg_classes.keys()}
        part_ious = {part: [] for part in seg_classes['Seafloor']}

        cur_precision, cur_recall, cur_f1 = 0.0, 0.0, 0.0

        classifier = classifier.eval()
        for batch_id, (points, label, target, file_name, point_set_normalized_mask, pc_min, pc_max) in \
                tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):

            print(batch_id)

            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes

            cur_pred = seg_pred.cpu().numpy()  # seg_pred.cpu().numpy()
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            cur_pred_prob = np.zeros((cur_batch_size, NUM_POINT)).astype(np.float64)
            target = target.cpu().data.numpy()
            point_set_normalized_mask = point_set_normalized_mask.numpy()
            cur_pred_val_mask = []
            cur_pred_prob_mask = []
            target_mask = []

            for i in range(cur_batch_size):
                # logits = cur_pred[i, :, :]
                prob = np.exp(cur_pred[i, :, :])
                # cur_pred_prob[i, :] = np.amax(prob, 1)
                cur_pred_prob[i, :] = prob[:, 1]  # the probability of belonging to seafloor class
                # cur_pred_val[i, :] = np.argmax(logits, 1)
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

                for i in range(cur_batch_size):
                    # mask out padded points
                    cur_points = points[i, :, :]
                    cur_mask = point_set_normalized_mask[i, :]
                    cur_points = cur_points[cur_mask, :]
                    # create a new point cloud array
                    output_points = np.zeros((cur_points.shape[0], 5)).astype(np.float64)
                    output_points[:, 0:3] = cur_points[:, 0:3]
                    # recover the point coordinates
                    cur_pc_min = pc_min[i, :]
                    cur_pc_max = pc_max[i, :]
                    output_points[:, 0:3] = pc_denormalize(output_points[:, 0:3], cur_pc_min, cur_pc_max)
                    # output class and probability
                    # output_points[:, 3] = cur_pred_val[i, :]
                    output_points[:, 3] = cur_pred_val_mask[i]
                    # output_points[:, 4] = cur_pred_prob[i, :]
                    output_points[:, 4] = cur_pred_prob_mask[i]
                    output_file = file_name[i]
                    output_path = os.path.join(output_dir, output_file)
                    np.savetxt(output_path, output_points, delimiter=' ', fmt='%.4f')

            for i in range(cur_batch_size):
                # segp = cur_pred_val[i, :]
                segp = cur_pred_val_mask[i]
                # segl = target[i, :].cpu().numpy()
                # segl = target[i, :]
                segl = target_mask[i]
                for l in [0, 1]:
                    # if (np.sum(segl == l) == 0) and (
                    #         np.sum(segp == l) == 0):  # part is not present, no prediction either
                    if np.sum(segl == l) == 0:
                        # part_ious[l - seg_classes[cat][0]] = 1.0
                        # part_ious2[l - seg_classes[cat][0]].append(1.0)
                        continue
                    else:
                        # iou = np.sum((segl == l) & (segp == l)) / float(
                        #     np.sum((segl == l) | (segp == l)))
                        iou = np.sum((segl == l) & (segp == l)) / float(
                            np.sum(segl == l))
                        part_ious[l].append(iou)
                        # part_ious2[l - seg_classes[cat][0]].append(part_ious[l - seg_classes[cat][0]])
                # shape_ious[cat].append(np.mean(part_ious))

            # seg_pred = seg_pred.reshape(-1, num_part)

            # target = torch.from_numpy(target).reshape(-1)
            # cur_pred_val = torch.from_numpy(cur_pred_val).reshape(-1)

            target_mask = np.hstack(target_mask)
            cur_pred_val_mask = np.hstack(cur_pred_val_mask)

            # calculate metric - F1 score
            # cm = confusion_matrix(seg_pred.cpu(), target.cpu(), num_classes=num_part)  # pytorch
            cm = confusion_matrix(target_mask, cur_pred_val_mask)  # sklearn
            # cm = cm.numpy()
            # accumulate true positives, false positives and false negatives
            # if no seafloor in both prediction and label
            if cm.shape[0] == 1:
                tp, fp, fn = 0, 0, 0
            else:
                # since we don't care about non-seafloor class, index start from 1
                tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]

            # accumulate tp, fp, fn
            tp_acc += tp
            fp_acc += fp
            fn_acc += fn

            # calculate batch-averaged precision and recall
            cur_precision += tp / (tp + fp) if (tp + fp) > 0 else 1.0
            cur_recall += tp / (tp + fn) if (tp + fn) > 0 else 1.0
            cur_f1 = 2 * cur_precision * cur_recall / (cur_precision + cur_recall) if (cur_precision + cur_recall) > 0 else 0.0

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

        # calculate the average of each batch
        precision_avg = cur_precision / len(testDataLoader)
        recall_avg = cur_recall / len(testDataLoader)
        f1_avg = cur_f1 / len(testDataLoader)
        log_string('Precision batch-averaged: %.5f' % precision_avg)
        log_string('Recall batch-averaged: %.5f' % recall_avg)
        log_string('F1 score batch-averaged: %.5f' % f1_avg)

        # all_shape_ious = []
        # for cat in shape_ious.keys():
        #     for iou in shape_ious[cat]:
        #         all_shape_ious.append(iou)
        #     shape_ious[cat] = np.mean(shape_ious[cat])
        # mean_shape_ious = np.mean(list(shape_ious.values()))
        mean_part_iou = []
        for part in sorted(part_ious.keys()):
            part_ious[part] = np.mean(part_ious[part])
            log_string('eval IoU of part %d: %f' % (part, part_ious[part]))
            mean_part_iou.append(part_ious[part])
        mean_part_iou = np.mean(mean_part_iou)
        test_metrics['part_avg_iou'] = mean_part_iou
        # test_metrics['class_avg_iou'] = mean_shape_ious
        # shape_iou = np.mean(shape_iou)
        # test_metrics['inctance_avg_iou'] = np.mean(shape_iou)

    log_string('Part avg mIOU is: %.5f' % test_metrics['part_avg_iou'])
    # log_string('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])
    # log_string('Inctance avg mIOU is: %.5f' % test_metrics['inctance_avg_iou'])


if __name__ == '__main__':
    args = parse_args()
    main(args)
