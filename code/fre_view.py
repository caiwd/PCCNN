import os
import random
from config import args

import matplotlib
matplotlib.use('Agg')
import numpy as np
import progressbar
from matplotlib import pyplot as plt
from scipy import signal

import _pickle as pickle

# plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 15
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.width'] =  2
plt.rcParams['ytick.major.width'] =  2


def decode_data(data):
    # alpha = 4*args.k/args.fs
    # b, a = signal.butter(4, alpha, 'highpass')
    # fx = signal.filtfilt(b, a, data)  # 高通滤波，<2000 Hz
    # hx = np.abs(signal.hilbert(fx))  # 希尔伯特变换
    # evop_data = np.sqrt(fx**2 + hx**2)
    # fre_data = np.abs(np.fft.rfft(evop_data))/len(evop_data)*2  # 包络FFT
    fre_data = np.abs(np.fft.rfft(data))/len(data)*2  # FFT
    return fre_data


def plot_fig(data, fre_data, fig_name='pic.png', per_fre=1, name_1='Time', name_2='Frequence'):
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(left=0.08, bottom=0.05, right=0.95,
                        top=0.95, wspace=0.2, hspace=0.2)
    plt.subplot(2, 1, 1)
    plt.plot(data)
    plt.title(name_1)
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(1, len(fre_data)+1)*per_fre, fre_data)
    plt.title(name_2)
    peak, _ = signal.find_peaks(
        fre_data, height=np.mean(fre_data), distance=10)
    plt.scatter((peak+1)*per_fre, [fre_data[x] for x in peak], c='r')
    # for x in peak:
    #     plt.text((x+2)*per_fre, fre_data[x], str((x+1)*per_fre))
    save_file_path = os.path.split(fig_name)[0]
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    plt.savefig(fig_name)
    # plt.show()
    plt.close()

def plot_fig_one(data, fre_data, fig_name='pic.png', per_fre=1, name_1='Time', name_2='Frequence'):
    plt.figure(figsize=(12, 5))
    plt.subplots_adjust(left=0.08, bottom=0.05, right=0.95,
                        top=0.95, wspace=0.2, hspace=0.2)
    plt.plot(np.arange(1, len(fre_data)+1)*per_fre, fre_data, lw=1.5)
    plt.title(name_2)
    peak, _ = signal.find_peaks(
        fre_data, height=np.mean(fre_data), distance=20)
    # plt.scatter((peak+1)*per_fre, [fre_data[x] for x in peak], c='r')
    for x in peak:
        plt.text((x+2)*per_fre, fre_data[x]*1.05, str((x+1)*per_fre))
    plt.xlim(0, len(fre_data))
    plt.ylim(0, max(fre_data)*1.08)
    save_file_path = os.path.split(fig_name)[0]
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    plt.savefig(fig_name)
    # plt.show()
    plt.close()

if __name__ == '__main__':
    # dataset_name = 'RMSF'
    dataset_name = 'CWRU'

    file_path = r'../data/{}/time/test'.format(dataset_name)
    names = [x for x in os.listdir(file_path) if '.pkl' in x]
    random.shuffle(names)
    try:
        os.makedirs(r'../imgs/{}/fre'.format(dataset_name))
    except OSError:
        pass
    pro_bar = progressbar.ProgressBar()
    for name in pro_bar(names[:200]):
        with open(os.path.join(file_path, name), 'rb') as f:
            data = pickle.load(f)
            data = np.reshape(data, (-1,))
            fre_data = decode_data(data)
            plot_fig(data, fre_data[1:1024], r'../img/{}/fre/{}.png'.format(
                dataset_name, name[:-4]), per_fre=1)
