import logging
import time

import os
import numpy as np
import torch
import torch.nn.functional as F

from config import args
from load_data import load_data
from Networks import AE
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from feature_save_plot import plot_tsne

plt.rcParams['font.family'] = 'Times New Roman'


def add_result(ori_dict, add_dict):
    for key in ori_dict.keys():
        ori_dict[key] = ori_dict[key] + add_dict[key]
    return ori_dict

def train_model(model, dataloader, optimizer, epoch):
    model.train()
    for datas, labels in dataloader:
        datas = torch.flatten(datas, 1)[:, :args.data_length]
        if torch.cuda.is_available() and args.use_cuda:
            datas = datas.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        _, output = model(datas)
        loss = torch.nn.MSELoss()(output, datas)
        loss.backward()
        optimizer.step()
    if epoch % args.log_interval==0:
        logging.warning('{}_{}   Epoch: {:<3} {:<8} Loss: {:.6f}'.format(args.tar_dataset_name, args.class_num, epoch, 'Train', loss))


def test_model(model, dataloader, epoch):
    model.eval()
    loss = 0
    for datas, labels in dataloader:
        datas = torch.flatten(datas, 1)[:, :args.data_length]
        if torch.cuda.is_available() and args.use_cuda:
            datas = datas.cuda()
            labels = labels.cuda()
        features, output = model(datas)
        datas = torch.flatten(datas, 1)
        features = torch.flatten(features, 1)
        output = torch.flatten(output, 1)
        loss = torch.nn.MSELoss()(output, datas)
        # break
        if torch.cuda.is_available() and args.use_cuda:
            datas = datas.cpu()
            labels = labels.cpu()
            features = features.cpu()
            output = output.cpu()
        features = features.detach().numpy()
        labels = labels.detach().numpy()
        datas = datas.detach().numpy()
        output = output.detach().numpy()

        try:
            all_features = np.concatenate((all_features, features), axis=0)
            all_labels = np.concatenate((all_labels, labels), axis=0)
            all_output = np.concatenate((all_output, output), axis=0)
            all_datas = np.concatenate((all_datas, datas), axis=0)
        except:
            all_features = features
            all_labels = labels
            all_output = output
            all_datas = datas

    if not os.path.exists(r'{}/AE'.format(args.img_path)):
        os.mkdir(r'{}/AE'.format(args.img_path))

    file_name = r'{}/AE/TSNE_{}_{}_epoch({})_class_num({})_experiment_time({}).png'.format(args.img_path, \
        args.tar_dataset_name, args.data_type, epoch, args.class_num, args.experiment_time)
    plot_tsne(all_features, all_labels, file_name, label_names=args.label_names)

    plt.figure(figsize=(16, 16))
    plt.suptitle('Loss: {:.3f}'.format(loss))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.065, right=0.965)
    for i in range(1, 17):
        plt.subplot(4, 4, i)
        plt.plot(all_output[i, :], label='Gen', alpha=0.75, color='r')
        plt.plot(all_datas[i, :], label='True', alpha=0.75, color='g')
        plt.title(args.label_names[int(all_labels[i])])
    plt.legend()
    if not os.path.exists('{}/{}'.format(args.img_path, 'AE')):
        os.makedirs('{}/{}'.format(args.img_path, 'AE'))
    plt.savefig(r'{}/AE/{}_{}_epoch({})_class_num({})_experiment_time({}).png'.format(args.img_path, \
        args.tar_dataset_name, args.data_type, epoch, args.class_num, args.experiment_time))
    plt.close()

    logging.warning('{}_{}   Epoch: {:<3} {:<8}   Loss: {:.6f}'.format(args.tar_dataset_name, args.class_num, epoch, 'Test ', loss/len(dataloader)))

    return loss


if __name__ == "__main__":
    if args.log_save:
        filename = r'{0}/{1}_AE_{2}.txt'.format(args.log_path, args.tar_dataset_name, time.strftime('%Y-%m-%d_%H-%M-%S'))
        logging.basicConfig(
            level=logging.WARNING, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s', filename=filename)
    else:
        logging.basicConfig(
            level=logging.WARNING, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    train_loader, test_loader, samples_num_list = load_data(
        args.tar_dataset_name, categories=args.class_num, num_workers=args.num_workers, normal_01=True)

    if args.data_type=='time' or args.data_type=='fre':
        model = AE('time_shao', args.data_length, args.class_num)
    elif args.data_type=='feature':
        model = AE('feature_li', args.data_length, args.class_num)
    if torch.cuda.is_available() and args.use_cuda:
        model.cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)

    acc_all = []
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1, last_epoch=-1)
    for epoch in range(1, args.epochs+1):
        train_model(model, train_loader, optimizer, epoch)
        scheduler.step()
        if epoch % args.log_interval==0:
            acc = test_model(model, test_loader, epoch)
            acc_all.append(acc)
    torch.save(model, r'{}/{}_{}_ae_model_class_num({})_experiment_time({}).pkl'.format(args.model_path, args.tar_dataset_name, args.data_type, args.class_num, args.experiment_time))
