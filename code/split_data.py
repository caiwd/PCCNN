# 将原始数据转换为ndarray数据保存，分为测试数据和训练数据
import os
import shutil
import time

import _pickle as pickle
import numpy as np
import pandas as pd
import logging
from nptdms import TdmsFile
from scipy.io import loadmat

from config import args
from fre_view import decode_data
from add_noise import awgn
from feature_save_plot import save_data_and_plot


def cwru(raw_path, save_path, label_names, length, space, split_rates, label_path, fre_time, decode=False, generalization=False):
    # decode为True表示存包洛解调后的频域数据
    if generalization:
        file_paths = [x for x in os.listdir(raw_path) if args.train_test in x]
    else:
        file_paths = [x for x in os.listdir(raw_path)]
    # print(file_paths)
    with open(os.path.join(label_path, 'label_{}_{}_{}.meta'.format(fre_time, os.path.split(raw_path)[-1], os.path.split(save_path)[-1])), 'w') as f:
        for file_path in file_paths:
            # 读取原始数据
            data_dict = loadmat(os.path.join(raw_path, file_path))
            DE_FE = 'DE'
            keys = [x for x in data_dict.keys() if '{}_time'.format(DE_FE)
                    in x][0]  # 取出驱动端传感器数据

            all_data = data_dict[keys]
            num = 0
            for split_rate in split_rates:
                useful_length = round(len(all_data)*(split_rate[1]-split_rate[0]))
                start = max(0, round(len(all_data)*split_rate[0])-length//2)
                stop = start + useful_length + length
                if stop > len(all_data)-1:
                    start = len(all_data) - useful_length - length
                    stop = len(all_data)
                if generalization:
                    data = all_data
                else:
                    data = all_data[start : stop]

                # 定义标签
                file_name = os.path.split(file_path)[-1][:-4]  # 获取文件名
                label = label_names.index(
                    [x for x in label_names if x in file_name][0])

                # 划分数据
                for i in range(0, len(data)-length, int(space)):  # 等间隔
                    one_sample = data[int(i):int(i)+length]
                    if args.add_awgn and args.train_test=='test':
                        one_sample = awgn(one_sample, args.awgn_db_list[args.experiment_time-1])
                    if decode:
                        one_sample = np.reshape(one_sample, (-1))
                        one_sample = decode_data(one_sample)[1:(args.fre_data_length+1)]

                    # 数据保存
                    path = r'{}/{}_{}.pkl'.format(save_path, file_name, num)
                    with open(path, 'wb') as data_f:
                        pickle.dump(one_sample, data_f)
                        f.write('{0}{1}{2}{3}'.format(path, '  ', label, '\n'))
                    num += 1
                if generalization:
                    break

    handle_label(label_names, label_path, 'label_{}_{}_{}.meta'.format(fre_time, os.path.split(raw_path)[-1], os.path.split(save_path)[-1]))  # 使每类的样本数量一致，并分别保存


def rmsf(raw_path, save_path, label_names, length, space, split_rates, label_path, fre_time, decode=False, generalization=False):
    # decode为True表示存包洛解调后的频域数据
    with open(os.path.join(label_path, 'label_{}_{}_{}.meta'.format(fre_time, os.path.split(raw_path)[-1], os.path.split(save_path)[-1])), 'w') as f:
        for label, label_name in enumerate(label_names):
            if generalization:
                label_paths_list = [x for x in os.listdir(
                    raw_path) if label_name in x and args.train_test in x]
            else:
                label_paths_list = [x for x in os.listdir(
                    raw_path) if label_name in x]
            # print(label_paths_list)
            num = 0
            for label_paths in label_paths_list:
                ori_file_paths = [x for x in os.listdir(os.path.join(
                    raw_path, label_paths)) if x[-5:] == '.tdms']
                files_num = len(ori_file_paths)
                
                for split_rate in split_rates:
                    if generalization:
                        file_paths = ori_file_paths
                    else:
                        file_paths = ori_file_paths[round(split_rate[0]*files_num): round(split_rate[1]*files_num)]
                    
                    for file_path in file_paths:
                        # 读取原始数据（单个）
                        file_path = os.path.join(os.path.join(
                            raw_path, label_paths), file_path)
                        tdms_file = TdmsFile(file_path)
                        try:
                            group_name, channel_name = 'group', 'acceleration'
                            datas = tdms_file[group_name][channel_name].data
                        except:
                            group_name, channel_name = '组名称', '加速度'
                            datas = tdms_file[group_name][channel_name].data

                        # 单个tdms文件数据
                        for i in range(0, len(datas)-length+1, space):
                            one_sample = datas[i:(i+length)]
                            if args.add_awgn and args.train_test=='test':
                                one_sample = awgn(one_sample, args.awgn_db_list[args.experiment_time-1])
                            if decode:
                                one_sample = np.reshape(one_sample, (-1))
                                one_sample = decode_data(one_sample)[1:(args.fre_data_length+1)]

                            # 数据保存
                            path = r'{}/{}_{}.pkl'.format(save_path, label_name, num)
                            with open(path, 'wb') as data_f:
                                pickle.dump(one_sample, data_f)
                                f.write('{0}{1}{2}{3}'.format(path, '  ', label, '\n'))
                            num += 1
                    if generalization:
                        break
    handle_label(label_names, label_path, 'label_{}_{}_{}.meta'.format(
        fre_time, os.path.split(raw_path)[-1], os.path.split(save_path)[-1]))  # 使每类的样本数量一致，并分别保存


def mfpt(raw_path, save_path, label_names, length, space, split_rates, label_path, fre_time, decode=False, generalization=False):
    # decode为True表示存包洛解调后的频域数据
    if generalization:
        file_paths = [x for x in os.listdir(raw_path) if args.train_test in x]
    else:
        file_paths = [x for x in os.listdir(raw_path)]
    # print(file_paths)
    with open(os.path.join(label_path, 'label_{}_{}_{}.meta'.format(fre_time, os.path.split(raw_path)[-1], os.path.split(save_path)[-1])), 'w') as f:
        for file_path in file_paths:
            # 读取原始数据
            raw_data = loadmat(os.path.join(raw_path, file_path))
            if 'Normal' in file_path:
                all_data = np.reshape(raw_data['bearing'][0][0][1], (-1,))
            else:
                all_data = np.reshape(raw_data['bearing'][0][0][2], (-1,))

            num = 0
            for split_rate in split_rates:
                useful_length = round(len(all_data)*(split_rate[1]-split_rate[0]))
                start = max(0, round(len(all_data)*split_rate[0])-length//2)
                stop = start + useful_length + length
                if stop > len(all_data)-1:
                    start = len(all_data) - useful_length - length
                    stop = len(all_data)
                if generalization:
                    data = all_data
                else:
                    data = all_data[start : stop]

                # 定义标签
                file_name = os.path.split(file_path)[-1][:-4]  # 获取文件名
                label = label_names.index(
                    [x for x in label_names if x in file_name][0])

                # 划分数据
                if label==0: # 将96k的采样频率降低为48k
                    data = data[range(1, len(data), 4)]
                else:
                    data = data[range(1, len(data), 2)]

                for i in range(0, len(data)-length, int(space)):  # 等间隔
                    one_sample = data[int(i):int(i)+length]
                    if args.add_awgn and args.train_test=='test':
                        one_sample = awgn(one_sample, args.awgn_db_list[args.experiment_time-1])
                    if decode:
                        one_sample = np.reshape(one_sample, (-1))
                        one_sample = decode_data(one_sample)[1:(args.fre_data_length+1)]

                    # 数据保存
                    path = r'{}/{}_{}.pkl'.format(save_path, file_name, num)
                    with open(path, 'wb') as data_f:
                        pickle.dump(one_sample, data_f)
                        f.write('{0}{1}{2}{3}'.format(path, '  ', label, '\n'))
                    num += 1
                if generalization:
                    break

    handle_label(label_names, label_path, 'label_{}_{}_{}.meta'.format(fre_time, os.path.split(raw_path)[-1], os.path.split(save_path)[-1]))  # 使每类的样本数量一致，并分别保存


def abvt(raw_path, save_path, label_names, length, space, split_rates, label_path, fre_time, decode=False, channel='C1', generalization=False):
    # decode为True表示存包洛解调后的频域数据
    with open(os.path.join(label_path, 'label_{}_{}_{}.meta'.format(fre_time, os.path.split(raw_path)[-1], os.path.split(save_path)[-1])), 'w') as f:
        for label, label_name in enumerate(label_names):
            if generalization:
                label_paths_list = [x for x in os.listdir(
                    raw_path) if label_name in x and args.train_test in x]
            else:
                label_paths_list = [x for x in os.listdir(
                    raw_path) if label_name in x]
            # print(label_paths_list)
            num = 0
            for label_paths in label_paths_list:
                ori_file_paths = [x for x in os.listdir(os.path.join(
                    raw_path, label_paths)) if x[-4:] == '.csv']
                for file_path in ori_file_paths:
                    # 读取原始数据（单个）
                    file_path = os.path.join(os.path.join(raw_path, label_paths), file_path)
                    raw_data = pd.read_csv(file_path, index_col=False, dtype=np.float32, names=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'])
                    all_data = np.array(raw_data[channel])
                    all_data = all_data[range(1, len(all_data), 2)] # 采样频率降低为25k Hz
                    for split_rate in split_rates:
                        useful_length = round(len(all_data)*(split_rate[1]-split_rate[0]))
                        start = max(0, round(len(all_data)*split_rate[0])-length//2)
                        stop = start + useful_length + length
                        if stop > len(all_data)-1:
                            start = len(all_data) - useful_length - length
                            stop = len(all_data)
                        if generalization:
                            datas = all_data
                        else:
                            datas = all_data[start : stop]
                        datas = datas[range(1, len(datas), 2)]

                        # 单个文件数据
                        for i in range(0, len(datas)-length+1, space):
                            one_sample = datas[i:(i+length)]
                            if args.add_awgn and args.train_test=='test':
                                one_sample = awgn(one_sample, args.awgn_db_list[args.experiment_time-1])
                            if decode:
                                one_sample = np.reshape(one_sample, (-1))
                                one_sample = decode_data(one_sample)[1:(args.fre_data_length+1)]

                            # 数据保存
                            path = r'{}/{}_{}.pkl'.format(save_path,
                                                            label_name, num)
                            with open(path, 'wb') as data_f:
                                pickle.dump(one_sample, data_f)
                                f.write('{0}{1}{2}{3}'.format(path, '  ', label, '\n'))
                            num += 1
                        if generalization:
                            break

    handle_label(label_names, label_path, 'label_{}_{}_{}.meta'.format(
        fre_time, os.path.split(raw_path)[-1], os.path.split(save_path)[-1]))  # 使每类的样本数量一致，并分别保存


def gb(raw_path, save_path, label_names, length, space, split_rates, label_path, fre_time, channel='C1', decode=False, generalization=False):
    # decode为True表示存包洛解调后的频域数据
    if generalization:
        file_paths = [x for x in os.listdir(raw_path) for y in label_names if y in x and args.train_test in x]
    else:
        file_paths = [x for x in os.listdir(raw_path) for y in label_names if y in x]
    # print(file_paths)
    with open(os.path.join(label_path, 'label_{}_{}_{}.meta'.format(fre_time, os.path.split(raw_path)[-1], os.path.split(save_path)[-1])), 'w') as f:
        for file_path in file_paths:
            # 读取原始数据
            try:
                datas = pd.read_csv(os.path.join(raw_path, file_path), dtype=np.float32, skiprows=range(0, 16), sep='\t', names=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
            except:
                datas = pd.read_csv(os.path.join(raw_path, file_path), dtype=np.float32, skiprows=range(0, 16), names=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])

            num = 0
            all_data = np.array(datas[channel])
            for split_rate in split_rates:
                useful_length = round(len(all_data)*(split_rate[1]-split_rate[0]))
                start = max(0, round(len(all_data)*split_rate[0])-length//2)
                stop = start + useful_length + length
                if stop > len(all_data)-1:
                    start = len(all_data) - useful_length - length
                    stop = len(all_data)
                if generalization:
                    data = all_data
                else:
                    data = all_data[start : stop]

                # 定义标签
                file_name = os.path.split(file_path)[-1][:-4]  # 获取文件名
                label = label_names.index(
                    [x for x in label_names if x in file_name][0])

                # 划分数据
                for i in range(0, len(data)-length, int(space)):  # 等间隔
                    one_sample = data[int(i):int(i)+length]
                    if args.add_awgn and args.train_test=='test':
                        one_sample = awgn(one_sample, args.awgn_db_list[args.experiment_time-1])
                    if decode:
                        one_sample = np.reshape(one_sample, (-1))
                        one_sample = decode_data(one_sample)[1:(args.fre_data_length+1)]

                    # 数据保存
                    path = r'{}/{}_{}.pkl'.format(save_path, file_name, num)
                    with open(path, 'wb') as data_f:
                        pickle.dump(one_sample, data_f)
                        f.write('{0}{1}{2}{3}'.format(path, '  ', label, '\n'))
                    num += 1
                if generalization:
                    break

    handle_label(label_names, label_path, 'label_{}_{}_{}.meta'.format(fre_time, os.path.split(raw_path)[-1], os.path.split(save_path)[-1]))  # 使每类的样本数量一致，并分别保存

def canda_wo(raw_path, save_path, label_names, length, space, split_rates, label_path, fre_time, decode=False, generalization=False):
    # decode为True表示存包洛解调后的频域数据
    if generalization:
        file_paths = [x for x in os.listdir(raw_path) if args.train_test in x]
    else:
        file_paths = [x for x in os.listdir(raw_path)]
    # print(file_paths)
    with open(os.path.join(label_path, 'label_{}_{}_{}.meta'.format(fre_time, os.path.split(raw_path)[-1], os.path.split(save_path)[-1])), 'w') as f:
        for file_path in file_paths:
            # 读取原始数据
            data_dict = loadmat(os.path.join(raw_path, file_path))
            all_data = data_dict['Channel_1']
            num = 0
            for split_rate in split_rates:
                useful_length = round(len(all_data)*(split_rate[1]-split_rate[0]))
                start = max(0, round(len(all_data)*split_rate[0])-length//2)
                stop = start + useful_length + length
                if stop > len(all_data)-1:
                    start = len(all_data) - useful_length - length
                    stop = len(all_data)
                if generalization:
                    data = all_data
                else:
                    data = all_data[start : stop]
                data = data[range(1, len(data), 8)]

                # 定义标签
                file_name = os.path.split(file_path)[-1][:-4]  # 获取文件名
                label = label_names.index(
                    [x for x in label_names if x in file_name][0])

                # 划分数据
                for i in range(0, len(data)-length, int(space)):  # 等间隔
                    one_sample = data[int(i):int(i)+length]
                    if args.add_awgn and args.train_test=='test':
                        one_sample = awgn(one_sample, args.awgn_db_list[args.experiment_time-1])
                    if decode:
                        one_sample = np.reshape(one_sample, (-1))
                        one_sample = decode_data(one_sample)[1:(args.fre_data_length+1)]

                    # 数据保存
                    path = r'{}/{}_{}.pkl'.format(save_path, file_name, num)
                    with open(path, 'wb') as data_f:
                        pickle.dump(one_sample, data_f)
                        f.write('{0}{1}{2}{3}'.format(path, '  ', label, '\n'))
                    num += 1
                if generalization:
                    break

    handle_label(label_names, label_path, 'label_{}_{}_{}.meta'.format(fre_time, os.path.split(raw_path)[-1], os.path.split(save_path)[-1]))  # 使每类的样本数量一致，并分别保存

def handle_label(label_names, label_path, name):
    # 生成标签数据
    with open(os.path.join(label_path, name), 'r') as f:
        import random
        all_labels = f.readlines()
        random.shuffle(all_labels)
        sample_num = []
        for _, label_name in enumerate(label_names):
            sample_num.append(len([x for x in all_labels if label_name in x]))
            exec('{}_list = []'.format(label_name))
        for line_data in all_labels:
            label_name = [x for x in label_names if x in line_data][0]
            exec('{}_list.append(line_data)'.format(label_name))
        exec('new_all_labels = []')
        min_sample_num = min(sample_num)  # 最小样本数目
        # logging.warning('{}_每类别的样本数目为：{}'.format(name, min_sample_num))
        for label_name in label_names:
            exec('temp_label_list = {0}_list[:{1}]'.format(
                label_name, min_sample_num))
            exec('new_all_labels.extend(temp_label_list)')

    for index, label_name in enumerate(label_names):  # 单个保存
        with open(os.path.join(label_path, '{}_{}.meta'.format(name[:-5], label_name)), 'w') as f:
            exec(
                'f.writelines(new_all_labels[index*min_sample_num:(index+1)*min_sample_num])')
    # 保存所有
    with open(os.path.join(label_path, '{}_same_num.meta'.format(name[:-5])), 'w') as f:
        exec('f.writelines(new_all_labels)')

if __name__ == '__main__':
    # 采用args.tar_dataset_name
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    experiment_time = args.experiment_time
    generalization = args.generalization
    split_rate_list = args.split_rate_list[min(9, experiment_time-1)]
    if args.add_awgn:
        split_rate_list = args.split_rate_list[0]

    if args.data_type == 'fre':  # 是否包洛解调
        is_decode = True
    else:
        is_decode = False

    for index, split_rates in enumerate(split_rate_list):  # 训练集与测试集的比例
        if index == 0:
            save_path = r'{}/{}/{}/{}/{}'.format(args.data_path, args.case,
                args.data_type, args.tar_dataset_name, 'train')
            args.train_test = 'train'
        else:
            save_path = r'{}/{}/{}/{}/{}'.format(args.data_path, args.case,
                args.data_type, args.tar_dataset_name, 'test')
            args.train_test = 'test'
        try:
            shutil.rmtree(save_path)  # 删除之前的数据文件夹
        except IOError:
            pass
        finally:
            os.makedirs(save_path)

        label_names = args.label_names
        length = args.data_length
        if args.tar_dataset_name in ['A', 'B', 'C']: # cwru数据
            cwru(raw_path=r'{}/{}/raw/{}'.format(args.data_path, args.case, args.tar_dataset_name), save_path=save_path, generalization=False,
                 label_names=label_names, length=length, space=args.data_space, decode=is_decode, split_rates=split_rates, label_path=args.label_path, fre_time=args.data_type)

        elif args.tar_dataset_name in ['D', 'E', 'F']: # 实验室数据
            rmsf(raw_path=r'{}/{}/raw/{}'.format(args.data_path, args.case, args.tar_dataset_name), save_path=save_path, generalization=generalization,
                 label_names=label_names, length=length, space=args.data_space, decode=is_decode, split_rates=split_rates, label_path=args.label_path, fre_time=args.data_type)

        elif args.tar_dataset_name in ['O', 'P']: # MFPT数据
            mfpt(raw_path=r'{}/{}/raw/{}'.format(args.data_path, args.case, args.tar_dataset_name), save_path=save_path, generalization=generalization,
                 label_names=label_names, length=length, space=args.data_space, decode=is_decode, split_rates=split_rates, label_path=args.label_path, fre_time=args.data_type)
        
        elif args.tar_dataset_name in ['G', 'H', 'I']: # 齿轮箱数据轴承
            gb(raw_path=r'{}/{}/raw/{}'.format(args.data_path, args.case, args.tar_dataset_name), save_path=save_path, channel='C7', generalization=False,
                 label_names=label_names, length=length, space=args.data_space, decode=is_decode, split_rates=split_rates, label_path=args.label_path, fre_time=args.data_type)

        elif args.tar_dataset_name in ['J', 'K', 'L', 'L_0']: # 齿轮箱齿轮数据
            gb(raw_path=r'{}/{}/raw/{}'.format(args.data_path, args.case, args.tar_dataset_name), save_path=save_path, channel='C7', generalization=False,
                 label_names=label_names, length=length, space=args.data_space, decode=is_decode, split_rates=split_rates, label_path=args.label_path, fre_time=args.data_type)

        elif args.tar_dataset_name in ['M', 'N', 'N_0']: # 12G数据
            abvt(raw_path=r'{}/{}/raw/{}'.format(args.data_path, args.case, args.tar_dataset_name), save_path=save_path, channel='C3', generalization=generalization,
                 label_names=label_names, length=length, space=args.data_space, decode=is_decode, split_rates=split_rates, label_path=args.label_path, fre_time=args.data_type)
        
        if args.tar_dataset_name in ['Q', 'R']: # 加拿大渥太华数据
            canda_wo(raw_path=r'{}/{}/raw/{}'.format(args.data_path, args.case, args.tar_dataset_name), save_path=save_path, generalization=(1-generalization),
                 label_names=label_names, length=length, space=args.data_space, decode=is_decode, split_rates=split_rates, label_path=args.label_path, fre_time=args.data_type)

        # logging.warning('{} 第{}次实验-数据集{}-{}-{}分割完成！'.format(time.strftime('%Y-%m-%d %H:%M:%S'), args.experiment_time, args.tar_dataset_name, args.data_type, args.train_test, ))
    
    save_data_and_plot(args, experiment_time)