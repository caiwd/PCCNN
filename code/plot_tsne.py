from config import args
from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from classify_method import get_features
from mpl_toolkits.mplot3d import Axes3D
import progressbar
from multiprocessing import Pool
import time

from load_data import load_data

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2

def plot_tsne_one_dataset(tar_dataset_name, args):
    class_num = args.class_num
    experiment_time = args.experiment_time
    args.anormal_method = 'gmm'
    color_list = ['r', 'g', 'b', 'k']
    if tar_dataset_name=='A':
        label_names = ['N', 'OR', 'IR', 'BF']
    elif tar_dataset_name=='B':
        label_names = ['N', 'OR', 'BF', 'IR']
    elif tar_dataset_name=='C':
        label_names = ['N', 'IR', 'BF', 'OR']
    elif tar_dataset_name=='D':
        label_names = ['N', 'OR', 'IR', 'MF']
    elif tar_dataset_name=='E':
        label_names = ['N', 'OR', 'MF', 'IR']
    elif tar_dataset_name=='F':
        label_names = ['N', 'IR', 'MF', 'OR']
    elif tar_dataset_name=='G':
        label_names = ['N', 'OR', 'IR', 'BF']
    elif tar_dataset_name=='H':
        label_names = ['N', 'OR', 'BF', 'IR']
    elif tar_dataset_name=='I':
        label_names = ['N', 'IR', 'BF', 'OR']
    elif tar_dataset_name=='J':
        label_names = ['N', 'OR', 'IR', 'BF']
    elif tar_dataset_name=='K':
        label_names = ['N', 'OR', 'BF', 'IR']
    elif tar_dataset_name=='L':
        label_names = ['N', 'IR', 'BF', 'OR']
    if class_num < 4:
        label_names[-1] = 'New'

    # 加载数据原始数据，cnn特征，mef特征
    _, time_test_loader, _ = load_data(tar_dataset_name, data_type='fre', categories=class_num, num_workers=0)
    time_datas, time_labels = get_features(time_test_loader)

    cnn_datas = np.loadtxt(r'{}/cnn_output_{}_{}_{}_{}.txt'.format(args.pccnn_output_path, tar_dataset_name, class_num, experiment_time,
    'gene' if args.generalization else 'noise' if args.add_awgn else 'baseline'))
    cnn_labels = np.loadtxt(r'{}/cnn_true_{}_{}_{}_{}.txt'.format(args.pccnn_output_path, tar_dataset_name, class_num, experiment_time,
    'gene' if args.generalization else 'noise' if args.add_awgn else 'baseline'))
    
    _, mef_test_loader, _ = load_data(tar_dataset_name, data_type='feature', categories=class_num, num_workers=0)
    mef_datas, mef_labels = get_features(mef_test_loader)

    # 绘图
    fig = plt.figure(figsize=(30, 8), dpi=100)
    for i, (datas, labels) in enumerate(zip([time_datas, mef_datas, cnn_datas], [time_labels, mef_labels, cnn_labels])):
        ax = plt.subplot(1, 3, i+1)
        # ax = fig.add_subplot(1, 3, i+1, projection='3d')
        datas = datas[:1000]
        labels = labels[:1000]
        clf = TSNE(n_components=2, perplexity=100)
        X_Y = clf.fit_transform(datas)

        label_done = []
        for index, i in enumerate(X_Y):
            color  = color_list[int(labels[index])]
            label_name = label_names[int(labels[index])]
            if label_name not in label_done:
                # ax.scatter(i[0], i[1], i[2], color=color, label=label_name)
                ax.scatter(i[0], i[1], color=color, label=label_name)
                label_done.append(label_name)
            else:
                # ax.scatter(i[0], i[1], i[2], color=color)
                ax.scatter(i[0], i[1], color=color)
        if index==2:
            plt.legend()
    plt.savefig(r'{}/TSNE_{}_experiment_time({}).png'.format(args.img_path, tar_dataset_name, experiment_time))
    # plt.show()
    plt.close()


if __name__ == "__main__":
    if args.class_num<4:
        tar_dataset_name_list = args.tar_dataset_name_list
        p = Pool(6)
        for tar_dataset_name in tar_dataset_name_list:
            p.apply_async(plot_tsne_one_dataset, args=(tar_dataset_name, args,))
            time.sleep(20)
        p.close()
        p.join()