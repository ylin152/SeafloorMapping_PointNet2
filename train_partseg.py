"""
Author: Benny
Date: Nov 2019

Modified by Yiwen Lin
Date: Jul 2023
"""
import argparse
import os, random
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import confusion_matrix
from pathlib import Path
from tqdm import tqdm
from data_utils.ShapeNetDataLoader import PartNormalDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    # if (y.is_cuda):
    #     return new_y.cuda()
    return new_y.to(y.device)
    # return new_y


def free_gpu_cache():
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_part_seg', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=251, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    parser.add_argument('--conf', action='store_true', default=False, help='use confidence level')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--data_root', type=str, required=True, help='data root file')
    parser.add_argument('--loss_weight', type=float, default=1.0, help='training loss weight')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='use early stopping or not')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''SEED SETTING'''
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    '''GPU SETTING'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # set GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    pretrain_dir = Path('')
    if args.ckpt:
        pretrain_dir = exp_dir.joinpath(args.ckpt)
    exp_dir = exp_dir.joinpath(timestr)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # tensorboard set-up
    writer = SummaryWriter(os.path.join('runs', timestr))

    root = args.data_root

    TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='train', conf_channel=args.conf)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=3, drop_last=True)
    VAL_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='val', conf_channel=args.conf)
    valDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=3)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of val data is: %d" % len(VAL_DATASET))

    num_classes = 1
    num_part = 2

    early_stopping = args.early_stopping

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.get_model(num_part, conf_channel=args.conf).to(device)
    # cross-entropy loss
    criterion = MODEL.get_loss().to(device)
    loss_weight = args.loss_weight
    weight = torch.Tensor([1.0, loss_weight]).to(device)
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(pretrain_dir))
        start_epoch = checkpoint['epoch']
        model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        classifier.load_state_dict(model_state_dict)
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0

    train_loss_acc_list = []
    val_loss_acc_list = []
    train_f1_acc_list = []
    val_f1_acc_list = []

    loss_dict = {"train_loss": [], "val_loss": []}
    metric_dict = {"train_metric": [], "val_metric": []}

    for epoch in range(start_epoch, args.epoch+start_epoch):

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch+start_epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        loss_acc = []
        # f1_acc = []
        tp, fp, fn = 0, 0, 0

        '''learning one epoch'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader),
                                                  smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            # data augmentation
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])

            points = torch.Tensor(points)
            # points [B,N,C]; label [B,num_classes]; target [B,N]
            points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))  # seg_pred: probabilities after softmax

            seg_pred = seg_pred.contiguous().view(-1, num_part)  # seg_pred [BxN, num_part]
            # seg_pred = seg_pred.contiguous().view(-1, 1)[:, 0]  # for sigmoid output
            target = target.view(-1, 1)[:, 0]  # target [BxN]
            # pred_choice = seg_pred.data.max(1)[1]
            # max(1) returns the maximum value along the axis 1 and its corresponding index
            # max(1)[0] returns the maximum value and max(1)[1] returns the index, which is the class number in our case

            # calculate confusion metric
            cm = confusion_matrix(seg_pred.detach(), target.detach(), num_classes=num_part)
            cm = cm.numpy()
            # accumulate true positives, false positives and false negatives
            # since we don't care about non-seafloor class, index start from 1
            tp += cm[1, 1]
            fp += cm[0, 1]
            fn += cm[1, 0]

            loss = criterion(seg_pred, target, weight=weight)
            # loss = criterion(seg_pred, target)
            loss_acc.append(loss.detach().item())
            loss.backward()
            optimizer.step()

        loss_acc = np.mean(loss_acc)
        train_loss_acc_list.append(loss_acc)
        log_string('Train loss: %.5f' % loss_acc)

        # calculate precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        # calculate recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        # calculate f1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        train_f1_acc_list.append(f1)
        log_string('Train F1 score: %.5f' % f1)

        val_loss_acc = []
        val_tp, val_fp, val_fn = 0, 0, 0

        # validation
        with torch.no_grad():
            for i, (points, label, target) in tqdm(enumerate(valDataLoader), total=len(valDataLoader),
                                                      smoothing=0.9):
                points = points.data.numpy()

                points = torch.Tensor(points)
                points, label, target = points.float().to(device), label.long().to(device), target.long().to(device)
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
                seg_pred = seg_pred.contiguous().view(-1, num_part)
                # seg_pred = seg_pred.contiguous().view(-1, 1)[:, 0]  # for sigmoid output
                target = target.view(-1, 1)[:, 0]
                # pred_choice = seg_pred.data.max(1)[1]

                # calculate confusion metric
                cm = confusion_matrix(seg_pred, target, num_classes=num_part)
                cm = cm.numpy()
                # accumulate true positives, false positives and false negatives
                # since we don't care about non-seafloor class, index start from 1
                val_tp += cm[1, 1]
                val_fp += cm[0, 1]
                val_fn += cm[1, 0]

                loss = criterion(seg_pred, target, weight=weight)
                # loss = criterion(seg_pred, target)
                val_loss_acc.append(loss.item())

        val_loss_acc = np.mean(val_loss_acc)
        val_loss_acc_list.append(val_loss_acc)
        log_string('Val loss: %.5f' % val_loss_acc)

        # calculate precision
        val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 1.0
        # calculate recall
        val_recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 1.0
        # calculate f1 score
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0.0
        val_f1_acc_list.append(val_f1)
        log_string('Val F1 score: %.5f' % val_f1)

        if epoch == start_epoch or (epoch + 1) % 10 == 0:
            # add scalar to tensorboard
            writer.add_scalar('Learning rate', lr, epoch + 1)
            writer.add_scalar('Loss/train', loss_acc, epoch + 1)
            writer.add_scalar('Loss/val', val_loss_acc, epoch + 1)
            writer.add_scalar('F1 score/train', f1, epoch + 1)
            writer.add_scalar('F1 score/val', val_f1, epoch + 1)
            writer.flush()
            writer.close()

        # early stopping
        if early_stopping is True:
            if len(val_loss_acc_list) > 5 and np.all(np.diff(val_loss_acc_list[-5:]) > 0):
                # save model if val loss doesn't improve for three epochs
                logger.info('Early Stopping...')
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch + 1,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

                free_gpu_cache()

                break

        # save checkpoints
        if (epoch + 1) % 50 == 0 and epoch != (args.epoch+start_epoch-1):
            logger.info('Save checkpoint at epoch %d...' % (epoch+1))
            save_ckptpath = str(checkpoints_dir) + '/ckpt_' + str(epoch + 1) + '.pth'
            log_string('Saving at %s' % save_ckptpath)
            state = {
                'epoch': epoch + 1,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, save_ckptpath)

        global_epoch += 1

        free_gpu_cache()

    logger.info('Save model...')
    savepath = str(checkpoints_dir) + '/model.pth'
    log_string('Saving at %s' % savepath)
    state = {
        'epoch': epoch + 1,
        # 'train_acc': train_instance_acc,
        # 'val_acc': val_instance_acc,
        'model_state_dict': classifier.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)

    logger.info('Test model...')
    # run test
    test_script = 'test_partseg.py'
    if args.conf:
        test_command = 'python ' + test_script + ' --num_point ' + str(args.npoint) + ' --batch_size ' \
                       + str(args.batch_size) + ' --log_dir ' + timestr + ' --data_root ' + args.data_root + ' --conf'
    else:
        test_command = 'python ' + test_script + ' --num_point ' + str(args.npoint) + ' --batch_size ' \
                   + str(args.batch_size) + ' --log_dir ' + timestr + ' --data_root ' + args.data_root
    return_code = os.system(test_command)
    if return_code != 0:
        print("Run test script error")


if __name__ == '__main__':
    args = parse_args()
    main(args)
