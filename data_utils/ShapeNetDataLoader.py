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
    def __init__(self, root = './data_8192', npoints=8192, split='train', class_choice=None, conf_channel=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category.txt')
        self.cat = {}
        self.split = split
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
        with open(os.path.join(self.root, 'train_test_split', 'train_file_list.json'), 'r') as f:
            train_ids = set([str(d) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'val_file_list.json'), 'r') as f:
            val_ids = set([str(d) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'test_file_list.json'), 'r') as f:
            test_ids = set([str(d) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
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
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

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
            point_set_coor = point_set_coor[choice]
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
            return point_set_normalized, cls, seg, \
                   file_name, point_set_normalized_mask, pc_min, pc_max, point_set_coor

        return point_set_normalized, cls, seg

    def __len__(self):
        return len(self.datapath)



