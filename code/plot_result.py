from config import args
import pandas as pd
import numpy as np
import os
import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 13
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] =  1.5
plt.rcParams['ytick.major.width'] =  1.5

args.tar_dataset_name_list.append('Aver')

def plot_compared_method():
    for tar_dataset_name in args.tar_dataset_name_list:
        datas = pd.read_excel(r'{}/更新前实验结果_不含标准差_{}_no_noise.xlsx'.format(args.result_path, 
                        'gene' if args.generalization else 'baseline'), index_col=0)
        for acc_type in ['known', 'unknown', 'aver']:
            plt.figure(figsize=(15, 7.5), dpi=100)
            ori_method_name = [x for x in datas.index if x!='feature_sp']
            acc_list = [datas.loc[x, '{}_{}'.format(tar_dataset_name, acc_type)] for x in ori_method_name]
            method_num = len(ori_method_name)
            plt.bar(range(method_num), acc_list, width=0.5)

            paper_method_name = ['CNN_'+x.split('_')[-1].upper() if 'cnn_feature' in x else 'MEF_'+x.split('_')[-1].upper() for x in ori_method_name]
            paper_method_name = [x if x!='CNN_SP' else 'CNN_SP (Proposed)' for x in paper_method_name]
            plt.xticks(range(method_num), paper_method_name, rotation=20)
            for a in range(method_num):
                b = datas.loc[ori_method_name[a], '{}_{}'.format(tar_dataset_name, acc_type)]
                plt.text(a, b+0.05, round(b, 2), ha='center', va= 'bottom')
            # plt.xlabel('方法名称')
            plt.ylabel('Accuracy / %')
            plt.title('Experimental results of different methods - {}_{}'.format(tar_dataset_name, acc_type))
            plt.savefig(r'{}/Experimental_results_of_different_methods_{}_no_noise_{}_{}.png'.format(args.img_path, 
                        'gene' if args.generalization else 'baseline', tar_dataset_name, acc_type))
            # plt.show()
            plt.close()


def plot_noise():
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    marks = ['o', '*', '+']
    beautiful_list = [x+y+'-' for x in marks for y in colors]
    for tar_dataset_name in args.tar_dataset_name_list:
        for acc_type in ['known', 'unknown', 'aver']:
            num = 0
            plt.figure(figsize=(10, 7.5), dpi=100)
            for method in args.anormal_methods_list:
                for feature_type in args.features_list:
                    if feature_type=='feature' and method=='sp':
                        continue
                    datas = pd.read_excel(r'{}/不同加噪比下的实验结果_{}_{}.xlsx'.format(args.result_path, 
                                'gene' if args.generalization else 'baseline', acc_type),
                                sheet_name=tar_dataset_name )
                    db_list = [args.awgn_db_list[x-1] for x in args.experiment_time_list]
                    plt.plot(db_list, datas['{}_{}'.format(feature_type, method)], beautiful_list[num], label='{}_{}{}'.format('MEF' if feature_type=='feature' else 'CNN', method.upper(), ' (Proposed)' if feature_type=='cnn_feature' and method=='sp' else ''))
                    num += 1
            plt.legend()
            plt.xlabel('SNR / DB')
            plt.ylabel('Accuracy / %')
            plt.title('Experimental results under different SNR - {}_{}'.format(tar_dataset_name, acc_type))
            plt.savefig(r'{}/Experimental_results_under_different_SNR_{}_{}_{}.png'.format(args.img_path, 
                        'gene' if args.generalization else 'baseline', tar_dataset_name, acc_type))
            # plt.show()
            plt.close()


def plot_update_result():
    from scipy import interpolate
    for tar_dataset_name in args.tar_dataset_name_list:
        for acc_type in ['known', 'new', 'aver']:
            plt.figure(figsize=(10, 7.5), dpi=100)
            for finetune in ['no_finetune', 'finetune']:
                for weight_loss in ['no_weight_loss', 'weight_loss']:
                    file_name = r'{}/不同新类别数目的实验结果_{}_{}_{}.xlsx'.format(args.result_path, 
                                'gene' if args.generalization else 'baseline',
                                finetune, weight_loss)
                    if os.path.exists(file_name):
                        datas = pd.read_excel((file_name), sheet_name=tar_dataset_name )
                        x = np.array(args.new_class_num_list)[:len(datas['{}'.format(acc_type)])]
                        x = np.log10(x)
                        y = datas['{}'.format(acc_type)]
                        plt.plot(x, y, 'o-', label='{}_{}{}'.format(finetune, weight_loss, ' (Proposed)' if finetune=='finetune' and weight_loss=='weight_loss' else ''))

                        # x_point = np.array([i for i in np.arange(min(x), max(x), (max(x)-min(x))/100)])
                        # func2 = interpolate.UnivariateSpline(x, y, s=100)
                        # plt.plot(x_point, [func2(x) for x in x_point], label='{}_{}'.format(finetune, weight_loss))
                        # plt.scatter(x, y)
            plt.legend()
            plt.xlabel('The sample number of new category / log10')
            plt.ylabel('Accuracy / %')
            plt.title('Experimental results for different sample numbers of new categories - {}_{}'.format(tar_dataset_name, acc_type))
            plt.savefig(r'{}/Experimental_results_under_different_sample_numbers_of_new_categories_{}_{}_{}.png'.format(args.img_path, 
                        'gene' if args.generalization else 'baseline', tar_dataset_name, acc_type))
            # plt.show()
            plt.close()


if __name__=='__main__':
    if args.task['plot_compared_method']:
        plot_compared_method()

    if args.task['plot_noise']:
        plot_noise()
    
    if args.task['plot_update_result']:
        plot_update_result()
    
    print('绘图完成')