import os
from multiprocessing import Pool

import numpy as np
import pywt
import shutil

import _pickle as pickle
from config import args
from split_data import handle_label
import logging
from feature_save_plot import save_data_and_plot


def extract_features_time(wave_data):
    p_mean = np.mean(wave_data)
    p_jfg = np.sqrt(np.mean(np.power(wave_data, 2)))
    p_std = np.std(wave_data)
    p_fgfz = np.mean(np.power(np.sqrt(np.abs(wave_data)), 2))
    p_abs_mean = np.mean(np.abs(wave_data))
    p_abs_max = np.max(np.abs(wave_data))
    p_max = np.max(wave_data)
    p_min = np.min(wave_data)
    p_p = p_max - p_min
    p_bxzb = p_jfg / p_abs_mean  # 波形指标
    p_fzzb = p_max / p_jfg  # 峰值指标
    p_mczb = p_max / p_abs_mean  # 脉冲指标
    p_ydzb = p_max / p_fgfz  # 裕度指标
    p_ptzb = np.mean(np.power((wave_data-p_mean)/p_std, 3))  # 偏态指标
    p_qdzb = np.mean(np.power((wave_data-p_mean)/p_std, 4))  # 俏度指标
    return [p_max, p_min, p_p, p_abs_max, p_fgfz, p_abs_mean, p_mean, p_std, p_jfg, p_bxzb, p_ptzb, p_qdzb, p_mczb, p_fzzb, p_ydzb]
    # return [p_p, p_mean, p_std, p_jfg, p_bxzb, p_ptzb, p_qdzb, p_mczb, p_fzzb, p_ydzb]


def extract_features_and_wavelet(parent_data_path, file_name, parent_save_path, is_wavelet=False):
    save_path = os.path.join(parent_save_path, file_name)
    with open(os.path.join(parent_data_path, file_name), 'rb') as f:
        wave_data = pickle.load(f)
        wave_data = np.reshape(wave_data, (-1, ))

    features = extract_features_time(wave_data)  # 时域特征

    x_point = np.diff(wave_data)/args.fs
    fre_c = np.sum(x_point*wave_data[1:]) / 2*np.pi*np.sum(np.power(wave_data, 2))
    fre_rmsf = np.sqrt( np.sum(np.power(x_point, 2)) / (4*np.pi**2*np.sum(np.power(wave_data, 2))) )
    fre_rvf = np.sqrt(np.abs(fre_rmsf**2-fre_c**2))
    features.extend([fre_c, fre_rmsf, fre_rvf])
    if is_wavelet:
        wp = pywt.WaveletPacket(data=wave_data, wavelet='db10', maxlevel=5)
        node_path = [x.path for x in wp.get_level(5, 'freq')]
        for node_name in node_path:
            features.append(
                np.mean(np.power(wp[node_name].data-np.mean(wp[node_name].data), 2)))
    with open(save_path, 'wb') as f:
        features = np.reshape(features, (-1, 1))
        pickle.dump(features, f)


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
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    Name = args.case
    dataset_name = args.tar_dataset_name
    experiment_time = args.experiment_time
    # logging.warning('经验特征提取[{}-18]...'.format(dataset_name))
    
    for train_test in ['train', 'test']:
        args.train_test = train_test
        save_label_path = args.label_path
        parent_data_path = '{}/{}/time/{}/{}'.format(args.data_path,
            Name, dataset_name, train_test)
        parent_save_path = '{}/{}/{}/{}/{}'.format(args.data_path,
            Name, args.data_type, dataset_name, train_test)
        try:
            shutil.rmtree(parent_save_path)
        except:
            pass
        finally:
            os.makedirs(parent_save_path)

        src_path = [x for x in os.listdir(parent_data_path) if '.pkl' in x]
        p = Pool(args.process_num)
        for file_name in src_path:
            p.apply_async(extract_features_and_wavelet, args=(parent_data_path, file_name, parent_save_path,))
            # extract_features_and_wavelet(parent_data_path, file_name, parent_save_path)
        p.close()
        p.join()

        gen_labels(dataset_name, parent_save_path, save_label_path, train_test)

    save_data_and_plot(args, experiment_time)
