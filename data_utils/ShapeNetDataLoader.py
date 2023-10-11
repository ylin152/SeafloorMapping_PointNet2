'''
Modified by Yiwen Lin
Date: Jul 2023
'''

# *_*coding:utf-8 *_*
import os, math
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
import torch
warnings.filterwarnings('ignore')


def pc_normalize(pc):
    pc_min = np.empty(3, dtype=np.float64)
    pc_max = np.empty(3, dtype=np.float64)
    for i in range(pc.shape[1]):
        pc_min[i] = min(pc[:, i])
        pc_max[i] = max(pc[:, i])
        pc[:, i] = 2 * ((pc[:, i] - pc_min[i]) / (pc_max[i] - pc_min[i])) - 1
    return pc, pc_min, pc_max


class PartNormalDataset(Dataset):
    def __init__(self, root='./data', npoints=8192, split='train', conf_channel=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.conf_channel = conf_channel

        with open(os.path.join(self.root, 'train_test_split', 'train_file_list.json'), 'r') as f:
            train_ids = set([str(d) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'val_file_list.json'), 'r') as f:
            val_ids = set([str(d) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'test_file_list.json'), 'r') as f:
            test_ids = set([str(d) for d in json.load(f)])

        self.datapath = []
        dir_point = os.path.join(self.root, 'input_data')
        fns = sorted(os.listdir(dir_point))
        if self.split == 'trainval':
            fns = [fn for fn in fns if ((fn in train_ids) or (fn in val_ids))]
        elif self.split == 'train':
            fns = [fn for fn in fns if fn in train_ids]  # fn[0:-4] to remove .txt if ids don't contain extension
        elif self.split == 'val':
            fns = [fn for fn in fns if fn in val_ids]
        elif self.split == 'test':
            fns = [fn for fn in fns if fn in test_ids]
        else:
            print('Unknown split: %s. Exiting..' % (split))
            exit(-1)
        for fn in fns:
            if os.path.splitext(os.path.basename(fn))[1] == '.txt':
                self.datapath.append(os.path.join(dir_point, fn))

        self.cache = {}
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = np.array([0]).astype(np.int32)
            data = np.loadtxt(fn).astype(np.float64)
            if not self.conf_channel:
                point_set = data[:, [0, 1, 2]]  # use x,y,elev
            else:
                point_set = data[:, [0, 1, 2, 6]]  # use x,y,elev,signal_conf
                point_set[:, -1] = point_set[:, -1].astype(np.int32)

            seg = data[:, -1].astype(np.int32)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)

        point_set_normalized = point_set
        point_set_normalized[:, 0:3], pc_min, pc_max = pc_normalize(point_set[:, 0:3])

        point_set_normalized_mask = np.full(self.npoints, True, dtype=bool)
        # resample
        if len(seg) > self.npoints:
            choice = np.random.choice(len(seg), self.npoints, replace=False)
            point_set_normalized = point_set_normalized[choice, :]
            seg = seg[choice]
        elif len(seg) < self.npoints:
            if not self.conf_channel:
                pad_point = np.ones((self.npoints-len(seg), 3), dtype=np.float32)
            else:
                pad_point = np.ones((self.npoints - len(seg), 3), dtype=np.float32)
                pad_conf = np.ones((self.npoints - len(seg), 1), dtype=np.int32)
                pad_point = np.concatenate((pad_point, pad_conf), axis=1)

            point_set_normalized = np.concatenate((point_set_normalized, pad_point), axis=0)

            # create mask for point set - mask out the padded points
            pad_point_bool = np.full(self.npoints - len(seg), False, dtype=bool)
            point_set_normalized_bool = np.full(len(seg), True, dtype=bool)
            point_set_normalized_mask = np.concatenate((point_set_normalized_bool, pad_point_bool))

            pad_seg = np.zeros(self.npoints-len(seg), dtype=np.int32)
            seg = np.concatenate((seg, pad_seg), axis=0)

        if self.split == 'test':
            return point_set_normalized, cls, seg, point_set_normalized_mask, pc_min, pc_max, fn

        return point_set_normalized, cls, seg

    def __len__(self):
        return len(self.datapath)



