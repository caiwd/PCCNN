import os
from multiprocessing import Pool

import numpy as np
import shutil
import torch

import torch.nn.functional as F

import _pickle as pickle
from config import args
from split_data import handle_label
import logging
from feature_save_plot import save_data_and_plot

from load_data import load_data


def model_predict(model, loader):
    model.eval()
    for index, (datas, labels) in enumerate(loader):
        datas = torch.flatten(datas, 1)[:, :args.data_length]
        if torch.cuda.is_available() and args.use_cuda:
            datas = datas.cuda()
        output, _  = model(datas)
        # p = F.softmax(output, dim=1)

        if torch.cuda.is_available() and args.use_cuda:
            output = output.cpu()
            labels = labels.cpu()
        labels = labels.detach().numpy()
        output = output.detach().numpy()

        if index != 0:
            all_output = np.concatenate((all_output, output), axis=0)
            all_labels = np.concatenate((all_labels, labels), axis=0)
        else:
            all_output = output
            all_labels = labels

    return all_output, all_labels


def save_pkl(features, save_path):
    with open(save_path, 'wb') as f:
        features = np.reshape(features, (-1, 1))
        pickle.dump(features, f)


def ae_features(model, loader, parent_save_path, tar_dataset_name, class_num):
    cnn_predict, true_lalbels = model_predict(model, loader)
    # p = Pool(int(os.cpu_count()*0.75))
    for i in range(len(true_lalbels)):
        label, features = true_lalbels[i], cnn_predict[i]
        file_name = args.label_names[label] + '_{}.pkl'.format(i)
        save_path = os.path.join(parent_save_path, file_name)
        # p.apply_async(save_pkl, args=(features, save_path, ))
        save_pkl(features, save_path)
    # p.close()
    # p.join()


def gen_labels(dataset_name, data_path, save_label_path, train_test):
    try:
        os.makedirs(save_label_path)
    except:
        pass
    label_names = args.label_names
    gen_label(data_path, save_label_path, label_names,
              label_name='label_{}_{}_{}.meta'.format(args.data_type, dataset_name, train_test))

    handle_label(label_names=label_names, label_path=save_label_path,
                 name='label_{}_{}_{}.meta'.format(args.data_type, dataset_name, train_test))


def gen_label(data_path, label_path, label_names, label_name):
    # 生成标签文件
    with open(r'{}/{}'.format(label_path, label_name), 'w') as f:
        all_file_names = [x for x in os.listdir(data_path) if '.pkl' in x]
        for file_name in all_file_names:
            label = label_names.index(
                [x for x in label_names if x in file_name][0])
            file_abso_path = os.path.join(data_path, file_name)
            f.write('{0}{1}{2}{3}'.format(file_abso_path, '  ', label, '\n'))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    case = args.case
    dataset_name = args.tar_dataset_name
    tar_dataset_name = args.tar_dataset_name
    experiment_time = args.experiment_time
    class_num = args.class_num
    input_data_type = 'fre'  # ae输入数据类型
    args.data_type = args.data_type  # ae输出数据类型
    # logging.warning('提取AE特征[{}-{}]...'.format(tar_dataset_name, class_num))

    model = torch.load(r'{}/{}_{}_ae_model_class_num({})_experiment_time({}).pkl'.format(args.model_path, tar_dataset_name,
        input_data_type, class_num, experiment_time), map_location=lambda storage, loc: storage)
    if torch.cuda.is_available() and args.use_cuda:
        model.cuda()

    all_labels_num = len(args.label_names)
    train_loader, test_loader, _ = load_data(
        tar_dataset_name, categories=all_labels_num, data_type='fre', num_workers=args.num_workers)

    for train_test in ['train', 'test']:
        args.train_test = train_test
        save_label_path = args.label_path
        parent_save_path = '{}/{}/{}/{}_{}'.format(args.data_path,
            case, args.data_type, tar_dataset_name, train_test)
        try:
            shutil.rmtree(parent_save_path)
        except:
            pass
        finally:
            os.makedirs(parent_save_path)

        if train_test == 'train':
            ae_features(model, train_loader, parent_save_path,
                         tar_dataset_name, class_num)
        elif train_test == 'test':
            ae_features(model, test_loader, parent_save_path,
                         tar_dataset_name, class_num)

        gen_labels(tar_dataset_name, parent_save_path,
                   save_label_path, train_test)

    save_data_and_plot(args, experiment_time)
