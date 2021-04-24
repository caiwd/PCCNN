import logging
import os
import platform
import shutil
import time

import numpy as np
import pandas as pd
import progressbar
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import args
from load_data import load_data
from Networks import Common_Net
from sfpt_copy import put
from write_excel import write2excel

one_hot_like_label = [[0.0/(args.class_num-1) if x != label else 1.0 for x in range(args.class_num)] if label < args.class_num else [
    1/args.class_num for x in range(args.class_num)] for label in range(args.class_num+1)]


class My_New_Loss(nn.Module):
    def __init__(self):
        super(My_New_Loss, self).__init__()
        return

    def forward(self, output, labels):
        new_labels = [one_hot_like_label[x] for x in labels]
        p = F.log_softmax(output, dim=1)
        new_labels = torch.tensor(new_labels)
        if torch.cuda.is_available() and args.use_cuda:
            new_labels = new_labels.cuda()
        all_loss = torch.sum(p*new_labels, dim=1)
        loss = -torch.mean(all_loss)

        return loss


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean'else loss.sum() if reduction == 'sum'else loss


def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)


def get_c_v(model='', loader='', p_s='', labels='', class_num=args.class_num):
    # 计算置信度向量
    if model != '':
        model.eval()
        for index, (data, label) in enumerate(loader):
            if torch.cuda.is_available() and args.use_cuda:
                data = data.cuda()
                label = label.cuda()
            p_temp = F.softmax(model(data), dim=1)

            if torch.cuda.is_available() and args.use_cuda:
                p_temp = p_temp.cpu()
                label = label.cpu()
            p_temp = p_temp.detach().numpy()
            label = label.detach().numpy()

            if index != 0:
                p_s = np.concatenate((p_s, p_temp), axis=0)
                labels = np.concatenate((labels, label), axis=0)
            else:
                p_s = p_temp
                labels = label

    c_v = []  # 置信度向量
    for label in range(class_num):
        condition_index = np.where(labels == label)[0]
        p = p_s[condition_index]
        p_temp = []
        for one_p in p:
            if max(one_p) == one_p[label]:
                p_temp.append(one_p)
        p_temp = np.array(p_temp)
        if len(np.shape(p_temp))==1:
            c = 1.0
        else:
            IQR = np.percentile(p_temp[:, label], 75) - np.percentile(p_temp[:, label], 25)
            c = np.percentile(p_temp[:, label], 25) - 1.5*IQR
        # c = max(np.mean(
        #     p_temp[:, label]) - args.threshold_std_times*np.std(p_temp[:, label]), 1/class_num)
        c_v.append(c)
    return c_v


def softmax_plus(p, labels, c_v, gen_data=False):
    result = {}  # 结果
    known_correct = []  # 已知类别正确数目
    unknown_correct = []  # 未知类别正确数目
    all_correct = []  # 总的正确数目
    for label in range(args.class_num):
        condition_index = torch.nonzero(labels == label, as_tuple=True)[0]
        p_temp = p[condition_index]

        c = c_v[label]
        correct_num = 0
        for item in p_temp:
            if item[label] >= c and item[label] == torch.max(item):
                correct_num += 1

        result[str(label)] = correct_num
        known_correct.append(correct_num)
        all_correct.append(correct_num)
    result['known'] = sum(known_correct)

    all_label_num = len(args.label_names)
    if gen_data:
        all_label_num = args.class_num + 1
    if args.class_num < all_label_num:  # 未知类别数据
        for i in range(all_label_num-args.class_num):
            unknown_category = args.class_num+i
            condition_index = torch.nonzero(
                labels == unknown_category, as_tuple=True)[0]
            p_temp = p[condition_index]
            correct_num = 0
            for item in p_temp:
                unknown_mark = 0
                for index, c in enumerate(c_v):
                    if item[index] < c:
                        unknown_mark += 1
                if unknown_mark == len(c_v):
                    correct_num += 1
            result[str(unknown_category)] = correct_num
            unknown_correct.append(correct_num)
            all_correct.append(correct_num)
        result['unknown'] = sum(unknown_correct)
    result['all'] = sum(all_correct)

    return result


def add_result(ori_dict, add_dict):
    # 将两个字典相同键的值累加
    for key in ori_dict.keys():
        ori_dict[key] = ori_dict[key] + add_dict[key]
    return ori_dict


def train_model(model, train_loader, test_loader, optimizer, epoch, weight_rate):
    my_new_loss = My_New_Loss()
    acc_all = []
    for index, (datas, labels) in enumerate(train_loader):
        model.train()
        if torch.cuda.is_available() and args.use_cuda:
            datas = datas.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        output = model(datas)
        # loss = my_new_loss(output, labels)
        loss = F.nll_loss(F.log_softmax(output, dim=1),
                          labels, weight=weight_rate)
        loss.backward()
        optimizer.step()

    return acc_all


def test_model(model, loader, c_v, epoch, test='Test', class_num=args.class_num, all_label_num=len(args.label_names)):
    model.eval()
    correct = 0
    loss = 0
    for datas, labels in loader:
        if torch.cuda.is_available() and args.use_cuda:
            datas = datas.cuda()
            labels = labels.cuda()
        output = model(datas)
        p = F.softmax(output, dim=1)
        try:
            result = add_result(result, softmax_plus(p, labels, c_v))
        except:
            result = softmax_plus(p, labels, c_v)
        # loss += F.nll_loss(F.log_softmax(output, dim=1), labels).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()  # 传统方法的准确率

    acc_cnn = correct/(len(loader.dataset)*class_num/all_label_num)
    acc_known = result['known']/(len(loader.dataset)*class_num/all_label_num)
    acc_all = result['all']/len(loader.dataset)
    if class_num == all_label_num:
        acc_unknown = 0
    else:
        acc_unknown = result['unknown'] / \
            (len(loader.dataset)*(1-class_num/all_label_num))

    result_str = '{:.2%}--{:.2%}--{:.2%}--{:.2%}'.format(
        acc_cnn, acc_known, acc_unknown, acc_all)
    for index in range(all_label_num):
        acc_temp = result[str(index)]/len(loader.dataset)*all_label_num
        result_str += '--{:.2%}'.format(acc_temp)

    logging.warning('Dataset：{}_{}   Epoch: {:<3} {:<8}   CNN_acc: {:.2%}   Known_acc: {:.2%}   Unknow_acc: {:.2%}   All_acc: {:.2%}'.format(
        args.tar_dataset_name, class_num, epoch, '{} '.format(test), acc_cnn, acc_known, acc_unknown, acc_all))

    return result_str, acc_all, acc_known


if __name__ == "__main__":
    method_name = os.path.splitext(os.path.basename(__file__))[0]
    log_filename = r'{}/{}_train_cnn_class_num({})_experiment_time({})_{}{}.txt'.format(
        args.log_path, args.tar_dataset_name, args.class_num, args.experiment_time, 
        'gene' if args.generalization else 'baseline', 
        '' if args.class_num!=len(args.label_names) else r'_{}_{}'.format(
        'finetune' if args.finetune else 'no_finetune',
        'weight_loss' if args.weight_loss else 'no_weight_loss'))
    model_filename = r'{}/{}_cnn_model_class_num({})_experiment_time({})_{}{}.pkl'.format(
        args.model_path, args.tar_dataset_name, args.class_num, args.experiment_time,
        'gene' if args.generalization else 'baseline', 
        '' if args.class_num!=len(args.label_names) else r'_{}_{}'.format(
        'finetune' if args.finetune else 'no_finetune', 
        'weight_loss' if args.weight_loss else 'no_weight_loss'))
    excel_filename = r'{}/{}_{}_experiment_time({})_{}{}.xlsx'.format(
        args.log_path, args.tar_dataset_name, method_name, args.experiment_time,
        'gene' if args.generalization else 'baseline', 
        '' if args.class_num!=len(args.label_names) else r'_{}_{}'.format(
        'finetune' if args.finetune else 'no_finetune', 
        'weight_loss' if args.weight_loss else 'no_weight_loss'))

    if args.log_save:
        logging.basicConfig(
            level=logging.WARNING, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s', filename=log_filename)
    else:
        logging.basicConfig(
            level=logging.WARNING, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    train_loader, test_loader, samples_num_list = load_data(
        args.tar_dataset_name, categories=args.class_num, num_workers=args.num_workers, normal_01=True)

    model = Common_Net(class_num=args.class_num, net_class=args.net_class)
    if torch.cuda.is_available() and args.use_cuda:
        model.cuda()

    params_groups = [{'params': model.model.sharedNet.conv_1.parameters(),
                      'lr': args.lr},
                     {'params': model.model.sharedNet.conv_1.parameters(),
                      'lr': args.lr*args.lr_rate},

                     {'params': model.model.sharedNet.bn_2.parameters(),
                      'lr': args.lr},
                     {'params': model.model.sharedNet.bn_2.parameters(),
                      'lr': args.lr*args.lr_rate},

                     {'params': model.model.sharedNet.conv_2.parameters(),
                      'lr': args.lr},
                     {'params': model.model.sharedNet.conv_2.parameters(),
                      'lr': args.lr*args.lr_rate},

                     {'params': model.model.sharedNet.bn_3.parameters(),
                      'lr': args.lr},
                     {'params': model.model.sharedNet.bn_3.parameters(),
                      'lr': args.lr*args.lr_rate},

                     {'params': model.model.sharedNet.conv_3.parameters(),
                      'lr': args.lr},
                     {'params': model.model.sharedNet.conv_3.parameters(),
                      'lr': args.lr*args.lr_rate},

                     {'params': model.model.sharedNet.bn_4.parameters(),
                      'lr': args.lr},
                     {'params': model.model.sharedNet.bn_4.parameters(),
                      'lr': args.lr*args.lr_rate},

                     {'params': model.model.sharedNet.conv_4.parameters(),
                      'lr': args.lr},
                     {'params': model.model.sharedNet.conv_4.parameters(),
                      'lr': args.lr*args.lr_rate},

                     {'params': model.model.bn_1_1.parameters(),
                      'lr': args.lr},
                     {'params': model.model.fc_1_1.parameters(),
                      'lr': args.lr},
                     {'params': model.model.bn_1_2.parameters(),
                      'lr': args.lr},
                     {'params': model.model.fc_1_2.parameters(),
                      'lr': args.lr},
                     ]

    params_groups_list = [[params_groups[y] for y in x] for x in
                          [[0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17], [1, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17],
                           [1, 3, 5, 6, 8, 10, 12, 14, 15, 16, 17], [
                               1, 3, 5, 7, 9, 10, 12, 14, 15, 16, 17],
                           [1, 3, 5, 7, 9, 11, 13, 14, 15, 16, 17]]]

    if args.class_num == len(args.label_names) and args.finetune:
        model_dict = model.state_dict()
        pre_model = torch.load(r'{}/{}_cnn_model_class_num({})_experiment_time({})_{}.pkl'.format(
                args.model_path, args.tar_dataset_name, args.class_num-1, args.experiment_time, 
                'gene' if args.generalization else 'baseline'))
        pre_model_dict = {k: v for k, v in pre_model.items(
        ) if k in model_dict and 'fc' not in k}
        model_dict.update(pre_model_dict)
        model.load_state_dict(model_dict)
        optimizer = torch.optim.RMSprop(
            params_groups_list[4], lr=args.lr, alpha=0.9)
    else:
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=args.lr, alpha=0.9)

    acc_all = []
    pro_bar = progressbar.ProgressBar()
    if args.class_num == len(args.label_names) and args.weight_loss:
        known_sample_num = samples_num_list[0]
        new_sample_num = samples_num_list[-1]
        new_rate = (known_sample_num/new_sample_num) / (known_sample_num/new_sample_num+args.class_num-1)
        known_rate = 1 / (known_sample_num/new_sample_num+args.class_num-1)
        weight_rate = [known_rate if index < (
            args.class_num-1) else new_rate for index in range(args.class_num)]
        weight_rate = torch.FloatTensor(weight_rate)
        if torch.cuda.is_available() and args.use_cuda:
            weight_rate = weight_rate.cuda()
    else:
        weight_rate = None

    log_interval = args.log_interval
    for epoch in range(1, args.epochs+1):
        train_model(model, train_loader, test_loader, optimizer, epoch, weight_rate)

        if epoch % log_interval == 0 or epoch==args.epochs:
            c_v = get_c_v(model, train_loader)
            _, _, acc_known = test_model(model, train_loader, c_v, epoch, all_label_num=args.class_num)

            if acc_known >= 0.99 or epoch==args.epochs:
                acc, _, _ = test_model(model, test_loader, c_v, epoch)
                acc_all.append(acc)
                logging.warning('Dataset：{}_{}   Epoch: {:<3} {:<8}   CNN_loss: {:.6f}   unknown_loss: {:.6f}'.format(
                    args.tar_dataset_name, args.class_num, epoch, 'Train', 1, 1))
                break

            if acc_known < 0.95 and epoch>5*args.log_interval:
                log_interval = 5*args.log_interval

    torch.save(model.state_dict(), model_filename)

    write2excel(excel_filename, pd.DataFrame(acc_all, columns=['sample_num({})_class_num({})'.format(samples_num_list[-1], args.class_num)]), '{}'.format(args.tar_dataset_name))

    if os.path.exists(args.backup_path):
        shutil.copyfile(log_filename, r'{}/{}'.format(args.backup_path, log_filename[3:]))
        shutil.copyfile(model_filename, r'{}/{}'.format(args.backup_path, model_filename[3:]))
        shutil.copyfile(excel_filename, r'{}/{}'.format(args.backup_path, excel_filename[3:]))
