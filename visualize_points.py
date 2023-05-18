import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/Users/evelyn/Desktop/PointNet2_pytorch/data17')
dir = '2023-05-08_19-09-56/output_ckpt_500'
export_dir = '2023-05-08_19-09-56/images_ckpt_500'
prob_flag = True

for file in os.listdir(dir):
    input_file = os.path.join('111', file)
    # ip_filename = os.path.splitext(os.path.basename(input_file))[0]
    ip_points = np.loadtxt(input_file).astype(np.float64)

    output_file = os.path.join(dir, file)
    op_points = np.loadtxt(output_file).astype(np.float64)

    fig = plt.figure(figsize=(10, 6))
    cdict = {0: 'olive', 1: 'royalblue'}
    label = {0: 'non-seafloor', 1: 'seafloor'}
    sf_annotated = ip_points[:, -1].astype(int)
    sf_annotated[sf_annotated == 2] = 0
    sf_predicted = op_points[:, 3]

    # set common x/y label
    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Elevation")

    # plot ground truth data
    ax1 = fig.add_subplot(1, 2, 1)
    for sf in np.unique(sf_annotated):
        i = np.where(sf_annotated == sf)
        ax1.scatter(ip_points[i, 1], ip_points[i, 4], s=0.05, c=cdict[sf], label=label[sf])
    ax1.set_title('Ground truth')

    ax2 = fig.add_subplot(1, 2, 2)
    if not prob_flag:
        # plot predicted data
        for sf in np.unique(sf_predicted):
            i = np.where(sf_predicted == sf)
            ax2.scatter(op_points[i, 1], op_points[i, 2], s=0.05, c=cdict[sf], label=label[sf])
        ax2.set_title('Prediction')
    else:
        # plot probability
        sf_prob = op_points[:, -1]
        cdict = {0: 'olive', 1: 'saddlebrown', 2: 'chocolate', 3: 'peru'}
        label = {0: 'non-seafloor', 1: 'high prob seafloor', 2: 'medium prob seafloor', 3: 'low prob seafloor'}
        high_prob_idx = np.where(sf_prob >= 0.8)
        ax2.scatter(op_points[high_prob_idx, 1], op_points[high_prob_idx, 2], s=0.05, c=cdict[1], label=label[1])
        medium_prob_idx = np.where((sf_prob >= 0.6) & (sf_prob < 0.8))
        ax2.scatter(op_points[medium_prob_idx, 1], op_points[medium_prob_idx, 2], s=0.05, c=cdict[2], label=label[2])
        low_prob_idx = np.where((sf_prob >= 0.4) & (sf_prob < 0.6))
        ax2.scatter(op_points[low_prob_idx, 1], op_points[low_prob_idx, 2], s=0.05, c=cdict[3], label=label[3])
        non_seafloor_idx = np.where(sf_prob < 0.4)
        ax2.scatter(op_points[non_seafloor_idx, 1], op_points[non_seafloor_idx, 2], s=0.05, c=cdict[0], label=label[0])


    handles, labels = ax1.get_legend_handles_labels()
    # plt.legend(handles, labels, loc='lower left', bbox_to_anchor=(1.04, 0.45))
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.11), ncol=2)
    plt.suptitle(file)

    plt.tight_layout()
    # plt.show()

    # save to png file
    if not os.path.exists(export_dir):
        os.mkdir(export_dir)
    export_file = os.path.join(export_dir, file+'.png')
    fig.savefig(export_file, dpi=200)

    plt.close()