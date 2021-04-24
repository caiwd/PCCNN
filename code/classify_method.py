from joblib.logger import PrintTime
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from multiprocessing import Pool, Manager
from sklearn.neighbors import LocalOutlierFactor
from svdd_base import SVDD
import os
from Networks import OCNN
import torch
import torch.nn.functional as F
import numpy as np
import logging
from config import args
from write_excel import write2excel
from load_data import load_data
import time
import pandas as pd
import warnings
import platform
import shutil

warnings.filterwarnings("ignore")

parameters = {"positive penalty": 0.9,
              "negative penalty": 0.8,
              "kernel": {"type": 'gauss'},
              "option": {"display": 'on'}}


def fit(model, fit_data, optimizer, epoch):
    # ocnn模型的训练
    model.train()
    for index in range(0, len(fit_data), args.batch_size):
        datas = fit_data[index:(index+args.batch_size)]
        datas = torch.tensor(datas)
        datas = torch.flatten(datas, 1)
        labels = torch.zeros(datas.size()[0], dtype=torch.long)
        if torch.cuda.is_available() and args.use_cuda:
            datas = datas.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        output = model(datas)
        loss = F.nll_loss(torch.sigmoid(output), labels)
        loss.backward()
        optimizer.step()
    logging.debug('Epoch: {:<3} {:<10} Loss: {:.6f}'.format(
        epoch, 'Train', loss))
    return model, optimizer


def score_samples(model, test_data, epoch=1):
    # ocnn模型的测试
    model.eval()
    for index in range(0, len(test_data), args.batch_size):
        datas = test_data[index:(index+args.batch_size)]
        datas = torch.tensor(datas)
        datas = torch.flatten(datas, 1)
        if torch.cuda.is_available() and args.use_cuda:
            datas = datas.cuda()
        output = model(datas)
        p = torch.sigmoid(output)
        if index != 0:
            scores = torch.cat((scores, p), dim=0)
        else:
            scores = p
    scores = torch.reshape(scores, (-1,))
    # logging.debug('Epoch: {:<3} {:<10} Loss: {:.6f}'.format(epoch, 'Train', loss))
    if torch.cuda.is_available() and args.use_cuda:
        scores = scores.cpu()

    return scores.detach().numpy()


def get_features(loader):
    for index, (datas, labels) in enumerate(loader):
        if args.data_type == 'time':
            datas = torch.flatten(datas, 1)

        if args.anormal_method == 'sp' or args.anormal_method == 'cnn':
            datas = F.softmax(datas, dim=2)[:, :, :args.class_num]

        datas = datas.detach().numpy()
        labels = labels.detach().numpy()
        datas = np.reshape(datas, (len(datas), -1))
        if index!=0:
            datas_all = np.concatenate((datas_all, datas), axis=0)
            labels_all = np.concatenate((labels_all, labels), axis=0)
        else:
            datas_all = datas
            labels_all = labels
        # logging.warning('训练样本数目：{}'.format(len(datas)))
    return datas_all, labels_all


def train(loader, epoch, model_list, method='ocsvm'):
    # 大于阈值表示属于正常
    # model_list 对需要多轮训练的模型有效， 传入上一次训练的模型，例如ocnn
    datas, labels = get_features(loader)

    threshold_list = []
    update_models = []
    update_optimizer = []
    clf_list, optimizers = model_list

    for label in range(args.class_num):  # 为每个类别拟合ocsvm模型
        condition_index = np.where(labels == label)[0]
        fit_data = datas[condition_index]  # 标签label的训练数据
        optimizer = optimizers[label]

        if method == 'ocsvm':
            clf = OneClassSVM()
        elif method == 'isofore':
            clf = IsolationForest()
        elif method == 'gmm':
            clf = BayesianGaussianMixture()
        elif method == 'svdd':
            clf = SVDD(parameters)
        elif method == 'lof':
            clf = LocalOutlierFactor(novelty=True, n_neighbors=int(fit_data.size*0.1))
        elif method == 'cnn':
            clf = ''
        elif method != 'sp':
            clf = clf_list[label]

        # 训练异常检测模型
        if method == 'ocnn':
            clf, optimizer = fit(clf, fit_data, optimizer, epoch)
            scores_temp = score_samples(clf, fit_data, epoch)
        elif method == 'lof':
            clf.fit(fit_data)
            scores_temp = clf.decision_function(fit_data)
        elif method == 'sp':
            pass
        elif method == 'cnn':
            pass
        else:
            clf.fit(fit_data)
            scores_temp = clf.score_samples(fit_data)

        # 异常检测模型阈值的计算
        if method != 'sp' and method != 'gmm' and method != 'cnn':
            threshold = np.mean(scores_temp) - \
                args.threshold_std_times*np.std(scores_temp)
            update_optimizer.append(optimizer)
            update_models.append(clf)
            threshold_list.append(threshold)
        elif method == 'gmm':
            threshold = np.mean(scores_temp)
            update_optimizer.append(optimizer)
            update_models.append(clf)
            threshold_list.append(threshold)
        elif method == 'sp':
            from cnn import get_c_v
            threshold_list = get_c_v(p_s=datas, labels=labels)
        elif method == 'cnn':
            threshold_list = ''

    model_list = (update_models, optimizers)
    return model_list, threshold_list


def test(loader, model_list, threshold_list, epoch, myself_threshold):
    datas, labels = get_features(loader)

    logging.warning('测试样本数目：{}'.format(len(datas)))
    all_label_num = len(args.label_names)
    correct_num_unknown = []
    correct_num_known = []
    correct_num_all = []
    result = {}
    clf_list, _ = model_list

    if args.anormal_method != 'sp' and args.anormal_method != 'cnn':
        for label in range(args.class_num):  # 已知类别的识别结果
            condition_index = np.where(labels == label)[0]
            test_data = datas[condition_index]  # 标签label的训练数据

            correct_index = []
            for clf, threshold in zip(clf_list, threshold_list):  # 用所有已知类别的异常检测模型依次进行检测
                if myself_threshold:
                    if args.anormal_method == 'ocnn':
                        test_scores = score_samples(clf, test_data)
                    elif args.anormal_method == 'lof':
                        test_scores = clf.decision_function(test_data)
                    else:
                        test_scores = clf.score_samples(test_data)
                    correct_index.append(
                        np.where(test_scores >= threshold, 1, 0))
                else:
                    correct_index.append(
                        np.where(clf.predict(test_data) == 1, 1, 0))
            correct_num = 0
            correct_index = np.array(correct_index)
            # 对每个样本逐个判断，for i in （样本的数目）
            for i in range(len(correct_index[0])):
                # 仅被真实标签的异常检测模型预测为正，其余均为负
                if np.sum(correct_index[:, i]) == 1 and correct_index[label, i] == 1:
                    correct_num += 1

            correct_num_known.append(correct_num)
            correct_num_all.append(correct_num)
            result[str(label)] = correct_num

        for i in range(all_label_num-args.class_num):  # 未知类别的识别结果
            unknown_category = args.class_num+i
            condition_index = np.where(labels == unknown_category)[0]
            test_data = datas[condition_index]  # 标签label的训练数据

            correct_num_list = []
            for clf, threshold in zip(clf_list, threshold_list):  # 用所有已知类别的异常检测模型依次进行检测
                if myself_threshold:
                    if args.anormal_method == 'ocnn':
                        scores_test = score_samples(clf, test_data)
                    else:
                        scores_test = clf.score_samples(test_data)
                    correct_num_temp = np.where(scores_test < threshold, 1, 0)
                else:
                    correct_num_temp = np.where(
                        clf.predict(test_data) == -1, 1, 0)
                correct_num_list.append(correct_num_temp)

            result_correct = np.sum(correct_num_list, axis=0)
            correct_num = len(np.where(result_correct == args.class_num)[0])

            correct_num_unknown.append(correct_num)
            correct_num_all.append(correct_num)
            result[str(unknown_category)] = correct_num

        result['known'] = np.sum(correct_num_known)
        result['unknown'] = np.sum(correct_num_unknown)
        result['all'] = np.sum(correct_num_all)

    elif args.anormal_method == 'sp':
        from cnn import softmax_plus
        c_v = threshold_list
        datas = torch.tensor(datas)
        labels = torch.tensor(labels)
        result = softmax_plus(datas, labels, c_v)

    elif args.anormal_method == 'cnn':
        known_correct_num = 0
        for label in range(args.class_num):  # 已知类别的识别结果
            condition_index = np.where(labels == label)[0]
            test_data = datas[condition_index]  # 标签label的数据
            num_temp = 0
            for i in test_data:
                if max(i) == i[label]:
                    num_temp += 1
                    known_correct_num += 1
            result[str(label)] = num_temp
        result['known'] = known_correct_num

        for i in range(all_label_num-args.class_num):  # 未知类别的识别结果
            unknown_category = args.class_num+i
            result[str(unknown_category)] = 0
        result['unknown'] = 0
        result['all'] = known_correct_num

    if args.class_num == all_label_num:
        acc_unknown = 0
    else:
        acc_unknown = result['unknown'] / \
            (len(loader.dataset)*(1-args.class_num/all_label_num))
    acc_known = result['known'] / \
        (len(loader.dataset)*args.class_num/all_label_num)
    acc_all = result['all']/len(loader.dataset)

    result_str = '{:.2%}--{:.2%}--{:.2%}'.format(
        acc_known, acc_unknown, acc_all)
    for index in range(all_label_num):
        acc_temp = result[str(index)]/len(loader.dataset)*all_label_num
        result_str += '--{:.2%}'.format(acc_temp)

    logging.warning('{:<27}  Epoch:{:<2} {:<4}  known_acc:{:0>6.2%}  unknow_acc:{:0>6.2%}  all_acc:{:0>6.2%}'.format('{}_{}_{}'.format(args.tar_dataset_name, args.anormal_method, args.data_type),
                                                                    epoch, 'Test', acc_known, acc_unknown, acc_all))

    return '{}'.format(result_str)


if __name__ == "__main__":
    if args.log_save:
        log_file_name = r'{}/{}_{}_{}_{}.txt'.format(args.log_path, args.tar_dataset_name, args.anormal_method,
                                'gene' if args.generalization else 'baseline',
                                'noise' if args.add_awgn else 'no_noise')
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s', filename=log_file_name)
    else:
        logging.basicConfig(
            level=logging.WARN, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    model_list = []
    optimizers = []
    method_name = args.anormal_method

    train_loader, test_loader, _ = load_data(args.tar_dataset_name, batch_size=args.batch_size, categories=args.class_num,
                                          data_type=args.data_type, normal_01=False, add_noise=False, num_workers=args.num_workers)
    # CNN
    if 'feature' in args.data_type and 'cnn' in args.data_type:
        data_type = 'cnn_feature'
    # AE
    elif 'feature' in args.data_type and 'cnn' not in args.data_type:
        data_type = 'ae_feature'
    # MF
    elif 'feature' in args.data_type and 'cnn' not in args.data_type:
        data_type = 'feature'

    for label in range(args.class_num):
        if args.anormal_method == 'ocnn':
            model = OCNN()
            optimizer = torch.optim.SGD(
                model.parameters(), lr=args.lr, momentum=args.momentum)
            if torch.cuda.is_available() and args.use_cuda:
                model.cuda()
            model_list.append(model)
            optimizers.append(optimizer)
        else:
            model_list.append([])
            optimizers.append([])
    model_list = (model_list, optimizers)

    acc_all = []
    for epoch in range(1, args.epochs+1):
        model_list, threshold_list = train(
            train_loader, epoch, method=args.anormal_method, model_list=model_list)
        if args.anormal_method == 'gmm' or args.anormal_method == 'ocnn':
            myself_threshold = True
        else:
            myself_threshold = False

        acc = test(test_loader, model_list, threshold_list, epoch, myself_threshold)
        acc_all.append(acc)

    logging.warning('{}_{}_{}_{}({})'.format(args.tar_dataset_name, method_name, data_type,
                                    'DB' if args.add_awgn else 'experiment_time',
                                    args.awgn_db_list[args.experiment_time-1] if args.add_awgn else args.experiment_time))

    result_file_name = r'{}/{}_{}_{}_{}({})_{}_{}.xlsx'.format(args.log_path, args.tar_dataset_name, data_type, method_name, 
                                'DB' if args.add_awgn else 'experiment_time',
                                args.awgn_db_list[args.experiment_time-1] if args.add_awgn else args.experiment_time,
                                'gene' if args.generalization else 'baseline',
                                'noise' if args.add_awgn else 'no_noise')

    write2excel(result_file_name, pd.DataFrame(list(acc_all), columns=['class_num({})'.format(args.class_num)]), '{}'.format(args.tar_dataset_name))

    if os.path.exists(args.backup_path):
        shutil.copyfile(result_file_name,
                        r'{}/{}'.format(args.backup_path, result_file_name[3:]))
