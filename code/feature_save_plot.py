from sklearn.manifold import TSNE
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import logging
from load_data import load_data
import os
import shutil

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] =  2.0
plt.rcParams['ytick.major.width'] =  2.0

def plot_tsne(data, label, file_name, label_names=None):
    # print('特征数据尺寸{}'.format(data.shape))
    data = data[:500]
    label = label[:500]
    clf = TSNE(n_components=2)
    X_Y = clf.fit_transform(data)

    color_list = []
    for index,i in enumerate(X_Y):
        color  = ['b', 'g', 'r', 'c', 'm', 'y', 'k'][int(label[index])]
        if color not in color_list:
            plt.scatter(i[0], i[1], color=color, label=label_names[int(label[index])])
            color_list.append(color)
        else:
            plt.scatter(i[0], i[1], color=color)
    plt.legend()

    plt.savefig(file_name)
    plt.close()

def plot_pic(data, label, save_path, args):
    for index in range(100):
        plt.figure(figsize=(10,6))
        plt.plot(data[index])
        plt.xlim(0, len(data[index]))
        if args.data_type=='fre':
            plt.ylim(0)
        plt.savefig('{}/{}_{}.png'.format(save_path, args.label_names[int(label[index])], index))
        plt.close()

def save_data_and_plot(args, experiment_time):
    if args.experiment_time_list[0]==experiment_time and False:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        logging.warning('保存特征数据并绘制TSNE图像...')
        train_loader, test_loader, _ = load_data(args.tar_dataset_name, num_workers=args.num_workers, data_type=args.data_type, categories=len(args.label_names), normal_01=True if args.data_type=='fre' else False, add_noise=False)
        for train_test, loader in zip(['train', 'test'] , [train_loader, test_loader]):
            for datas, labels in loader:
                try:
                    all_datas = torch.cat((all_datas, datas), dim=0)
                    all_labels = torch.cat((all_labels, labels), dim=0)
                except:
                    all_datas = datas
                    all_labels = labels
            all_datas = all_datas.detach().numpy()
            all_labels = all_labels.detach().numpy()
            all_datas = np.reshape(all_datas, (len(all_datas), -1))
            all_labels = np.reshape(all_labels, (len(all_labels), -1))

            np.savetxt(r'{}/{}/{}/datas_{}_{}.txt'.format(args.data_path, args.case, args.data_type, args.tar_dataset_name, train_test), all_datas)
            np.savetxt(r'{}/{}/{}/labels_{}_{}.txt'.format(args.data_path, args.case, args.data_type, args.tar_dataset_name, train_test), all_labels)

            save_img_path = r'{}/{}/{}'.format(args.img_path, args.case, args.data_type)
            save_img_path_1 = r'{}/{}/{}/{}/{}'.format(args.img_path, args.case, args.data_type, args.tar_dataset_name, train_test)
            
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)

            try:
                shutil.rmtree(save_img_path_1)
            except:
                pass
            finally:
                os.makedirs(save_img_path_1)

            if args.experiment_time == args.experiment_time_list[0]:
                plot_pic(all_datas, all_labels, '{}'.format(save_img_path_1), args)

            # plot_tsne(all_datas, all_labels, r'{}/{}_{}_{}.png'.format(save_img_path, args.tar_dataset_name, train_test, args.experiment_time))