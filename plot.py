import os
import numpy as np
from matplotlib import pyplot as plt

dir = 'data17/2023-05-08_19-09-56/'
file = os.path.join(dir, 'pointnet2_part_seg_msg2_3_3.txt')
loss_flag = True
f1_score_flag = True

# plot loss
if loss_flag:
    train_loss_list = []
    val_loss_list = []
    with open(file, 'r') as f_obj:
        for line in f_obj:
            if 'Train loss' in line:
                train_loss_list.append(float(line.split(':')[3].strip()))
            if 'Val loss' in line:
                val_loss_list.append(float(line.split(':')[3].strip()))

    fig = plt.figure(figsize=(10, 6))
    plt.plot(train_loss_list[:600], label='Training loss')
    plt.plot(val_loss_list[:600], label='Validation loss')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(dir, 'loss.png'))

# plot f1-score
if f1_score_flag:
    train_f1_list = []
    val_f1_list = []
    with open(file, 'r') as f_obj:
        for line in f_obj:
            if 'Train F1 score' in line:
                train_f1_list.append(float(line.split(':')[3].strip()))
            if 'Val F1 score' in line:
                val_f1_list.append(float(line.split(':')[3].strip()))

    fig = plt.figure(figsize=(10, 6))
    plt.plot(train_f1_list[:600], label='Training f1 score')
    plt.plot(val_f1_list[:600], label='Validation f1 score')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(dir, 'F1 score.png'))