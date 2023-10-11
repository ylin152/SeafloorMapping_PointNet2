"""
Author: Benny
Date: Nov 2019

Modified by Yiwen Lin
Date: Jul 2023
"""
import argparse
import os
from pathlib import Path
from data_utils.ShapeNetDataLoader import PartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
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
    # set GPUs
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
            output_dir = Path(experiment_dir + '/output_' + str(args.ckpt).split('.')[0] + '_' + str(args.threshold))
        else:
            output_dir = Path(experiment_dir + '/output_' + str(args.threshold))

        if not os.path.exists(output_dir):
            output_dir.mkdir()

    root = args.data_root

    TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', conf_channel=args.conf)
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

    classifier.load_state_dict(checkpoint['model_state_dict'])

    thres = args.threshold

    with torch.no_grad():

        tp_acc, fp_acc, fn_acc = 0, 0, 0

        test_metrics = {}

        part_ious = {part: [] for part in seg_classes['Seafloor']}

        classifier = classifier.eval()
        for batch_id, (points, label, target, point_set_normalized_mask, pc_min, pc_max, fn) in \
                tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):

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

                for i in range(cur_batch_size):
                    # mask out padded points
                    cur_points = points[i, :, :]
                    cur_mask = point_set_normalized_mask[i, :]
                    cur_points = cur_points[cur_mask, :]
                    # create a new point cloud array
                    output_points = np.zeros((cur_points.shape[0], 7)).astype(np.float64)
                    # recover the point coordinates
                    output_points[:, 0:3] = cur_points[:, 0:3]
                    cur_pc_min = pc_min[i, :]
                    cur_pc_max = pc_max[i, :]
                    # recover other information
                    data = np.loadtxt(fn[i]).astype(np.float64)
                    cur_lonlat = data[:, [3, 4]]
                    # output points
                    output_points[:, 0:3] = pc_denormalize(output_points[:, 0:3], cur_pc_min, cur_pc_max)
                    # output lon/lat
                    output_points[:, 3:5] = cur_lonlat
                    # output class and probability
                    output_points[:, 5] = cur_pred_prob_mask[i]
                    output_points[:, 6] = cur_pred_val_mask[i]
                    # output file
                    output_file = os.path.basename(fn[i])
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
            # accumulate true positives, false positives and false negatives
            # if batch size = 1 (and negative sample), and there are no false positives, then set tp, fp, fn to 0
            if cm.shape[0] == 1:
                tp, fp, fn = 0, 0, 0
            else:
                # since we don't care about non-seafloor class, index start from 1
                tp, fp, fn = cm[1, 1], cm[0, 1], cm[1, 0]

            # accumulate tp, fp, fn
            tp_acc += tp
            fp_acc += fp
            fn_acc += fn

        # calculate globally on the entire test dataset
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


if __name__ == '__main__':
    args = parse_args()
    main(args)
