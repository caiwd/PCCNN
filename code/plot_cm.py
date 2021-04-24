from enum import EnumMeta
from sklearn.metrics import confusion_matrix
from config import args
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from plot_roc import get_label_names

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 30
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] =  1.5
plt.rcParams['ytick.major.width'] =  1.5
plt.rcParams['figure.figsize'] =  (30, 10)

def load_pre_true_label(tar_dataset_name, class_nums, experiment_time):
    all_labels = []
    for class_num, tar_dataset_name in zip(class_nums, tar_dataset_name):
        pre_labels = np.loadtxt(r'{}/cnn_pre_{}_class_num({})_experiment_time({})_{}.txt'.format(
                    args.cnn_output_path, tar_dataset_name, class_num, args.experiment_time, 
                    'gene' if args.generalization else 'noise' if args.add_awgn else 'baseline'))
        true_labels = np.loadtxt(r'{}/cnn_true_{}_class_num({})_experiment_time({})_{}.txt'.format(
                    args.cnn_output_path, tar_dataset_name, class_num, args.experiment_time, 
                    'gene' if args.generalization else 'noise' if args.add_awgn else 'baseline'))
        all_labels.append([pre_labels, true_labels])

    return all_labels

def tuple_round(CM):
    for index_x, x in enumerate(CM):
        for index_y, y in enumerate(x):
            CM[index_x][index_y] = round(y, 2)
    return CM

def plot_cm(all_labels, dataset_names, fig_name, class_nums, col=3):
    plt.figure(figsize=(col*12, 8), dpi=100)
    for i in range(len(dataset_names)):
        class_num = class_nums[i]
        pre_labels = all_labels[i][0]
        true_labels = all_labels[i][1]
        _, label_names = get_label_names(dataset_names[i])
        label_names[-1] = 'New'
        CM = confusion_matrix(true_labels, pre_labels, labels=list(range(0, class_num+1)), normalize='true')
        CM = tuple_round(CM)
        ax = plt.subplot(1, col, i+1)
        ax = sns.heatmap(CM, cmap="YlGnBu", linecolor='y', cbar=True, annot=True, linewidths=.5, ax=ax, xticklabels=label_names, yticklabels=label_names)

    plt.savefig(fig_name)
    # plt.show()
    plt.close()

if __name__=='__main__':
    experiment_time = args.experiment_time

    dataset_names = ['A', 'B', 'C']
    class_nums = [3, 3, 3]
    all_labels = load_pre_true_label(dataset_names, class_nums, experiment_time)
    plot_cm(all_labels, dataset_names, r'{}/CM_CWRU_experiment_time({}).png'.format(args.img_path, experiment_time), class_nums)

    dataset_names = ['D', 'E', 'F']
    class_nums = [3, 3, 3]
    all_labels = load_pre_true_label(dataset_names, class_nums, experiment_time)
    plot_cm(all_labels, dataset_names, r'{}/CM_RMSF_experiment_time({}).png'.format(args.img_path, experiment_time), class_nums)

    dataset_names = ['G', 'H', 'I']
    class_nums = [3, 3, 3]
    all_labels = load_pre_true_label(dataset_names, class_nums, experiment_time)
    plot_cm(all_labels, dataset_names, r'{}/CM_GB_Bearing_experiment_time({}).png'.format(args.img_path, experiment_time), class_nums)

    dataset_names = ['J', 'K', 'L', 'L_0']
    class_nums = [4, 4, 4, 4]
    all_labels = load_pre_true_label(dataset_names, class_nums, experiment_time)
    plot_cm(all_labels, dataset_names, r'{}/CM_GB_Gearbox_experiment_time({}).png'.format(args.img_path, experiment_time), class_nums, col=4)

    dataset_names = ['O', 'P']
    class_nums = [2, 2]
    all_labels = load_pre_true_label(dataset_names, class_nums, experiment_time)
    plot_cm(all_labels, dataset_names, r'{}/CM_MFPT_experiment_time({}).png'.format(args.img_path, experiment_time), class_nums, col=2)

    dataset_names = ['Q', 'R']
    class_nums = [2, 2]
    all_labels = load_pre_true_label(dataset_names, class_nums, experiment_time)
    plot_cm(all_labels, dataset_names, r'{}/CM_UO_experiment_time({}).png'.format(args.img_path, experiment_time), class_nums, col=2)

    dataset_names = ['M', 'N', 'N_0']
    class_nums = [3, 3, 3]
    all_labels = load_pre_true_label(dataset_names, class_nums, experiment_time)
    plot_cm(all_labels, dataset_names, r'{}/CM_ABVT_experiment_time({}).png'.format(args.img_path, experiment_time), class_nums)