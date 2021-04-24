from config import args
from matplotlib import pyplot as plt
import numpy as np
from write_excel import write2excel
import pandas as pd

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 25
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.width'] =  2
plt.rcParams['ytick.major.width'] =  2

def get_label_names(tar_dataset_name):
    if tar_dataset_name=='A':
        case = 'CWRU'
        label_names = ['N', 'OR', 'IR', 'BRF']
    elif tar_dataset_name=='B':
        case = 'CWRU'
        label_names = ['N', 'OR', 'BRF', 'IR']
    elif tar_dataset_name=='C':
        case = 'CWRU'
        label_names = ['N', 'IR', 'BRF', 'OR']

    elif tar_dataset_name=='D':
        case = 'RMSF'
        label_names = ['N', 'OR', 'IR', 'MIS']
    elif tar_dataset_name=='E':
        case = 'RMSF'
        label_names = ['N', 'OR', 'MIS', 'IR']
    elif tar_dataset_name=='F':
        case = 'RMSF'
        label_names = ['N', 'IR', 'MIS', 'OR']
        
    elif tar_dataset_name=='G':
        case = 'SU_B'
        label_names = ['N', 'OR', 'IR', 'BRF']
    elif tar_dataset_name=='H':
        case = 'SU_B'
        label_names = ['N', 'OR', 'BRF', 'IR']
    elif tar_dataset_name=='I':
        case = 'SU_B'
        label_names = ['N', 'IR', 'BRF', 'OR']

    elif tar_dataset_name=='J':
        case = 'SU_G'
        label_names = ['N', 'CT', 'MS', 'RC', 'SW']
    elif tar_dataset_name=='K':
        case = 'SU_G'
        label_names = ['N', 'CT', 'MS', 'SW', 'RC']
    elif tar_dataset_name=='L':
        case = 'SU_G'
        label_names = ['N', 'CT', 'SW', 'RC', 'MS']
    elif tar_dataset_name=='L_0':
        case = 'SU_G'
        label_names = ['N', 'SW', 'MS', 'RC', 'CT']

    elif tar_dataset_name=='M':
        case = 'ABVT'
        label_names = ['N', 'MIS', 'IM', 'BF']
    elif tar_dataset_name=='N':
        case = 'ABVT'
        label_names = ['N', 'MIS', 'BF', 'IM']
    elif tar_dataset_name=='N_0':
        case = 'ABVT'
        label_names = ['N', 'BF', 'IM', 'MIS']

    elif tar_dataset_name=='O':
        case = 'MFPT'
        label_names = ['N', 'OR', 'IR']
    elif tar_dataset_name=='P':
        case = 'MFPT'
        label_names = ['N', 'IR', 'OR']
    
    elif tar_dataset_name=='Q':
        case = 'UO'
        label_names = ['N', 'OR', 'IR']
    elif tar_dataset_name=='R':
        case = 'UO'
        label_names = ['N', 'IR', 'OR']
    
    return case, label_names

if __name__=='__main__':
    experiment_time = args.experiment_time
    for case_list in [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I'], ['J', 'K', 'L', 'L_0'], ['M', 'N', 'N_0'], ['O', 'P'], ['Q', 'R']]:
        plt.figure(figsize=(len(case_list)*10, 8), dpi=100)
        for i, tar_dataset_name in enumerate(case_list):
            plt.subplot(1, len(case_list), i+1)
            case, label_names = get_label_names(tar_dataset_name)

            class_num = len(label_names)-1

            p = np.loadtxt(r'{}/cnn_output_{}_class_num({})_experiment_time({})_{}.txt'.format(
                    args.cnn_output_path, tar_dataset_name, class_num, args.experiment_time, 
                    'gene' if args.generalization else 'noise' if args.add_awgn else 'baseline'))
            pre_labels = np.loadtxt(r'{}/cnn_pre_{}_class_num({})_experiment_time({})_{}.txt'.format(
                    args.cnn_output_path, tar_dataset_name, class_num, args.experiment_time, 
                    'gene' if args.generalization else 'noise' if args.add_awgn else 'baseline'))
            true_labels = np.loadtxt(r'{}/cnn_true_{}_class_num({})_experiment_time({})_{}.txt'.format(
                    args.cnn_output_path, tar_dataset_name, class_num, args.experiment_time, 
                    'gene' if args.generalization else 'noise' if args.add_awgn else 'baseline'))
            excel_filename_roc = r'{}/roc_experiment_time({})_{}.xlsx'.format(
                    args.result_path, args.experiment_time, 
                    'gene' if args.generalization else 'noise' if args.add_awgn else 'baseline')
            excel_filename_pr = r'{}/pr_experiment_time({})_{}.xlsx'.format(
                    args.result_path, args.experiment_time, 
                    'gene' if args.generalization else 'noise' if args.add_awgn else 'baseline')

            color_list = ['r', 'g', 'b', 'k']
            for label in range(class_num):
                unknown_index = np.where(true_labels==class_num)[0]
                unknown_score = p[unknown_index][:, label]
                unknow_label = np.zeros(shape=len(unknown_index))

                known_index = np.where(true_labels==label)[0]
                known_score = p[known_index][:, label]
                known_label = np.ones(shape=len(known_index))

                true_labels_temp = np.concatenate((unknow_label, known_label), axis=0)
                y_score_temp = np.concatenate((unknown_score, known_score), axis=0)

                fpr, tpr, threshold = roc_curve(true_labels_temp, y_score_temp, drop_intermediate=False) ### 计算真正率和假正率
                precision, recall, threshold = precision_recall_curve(true_labels_temp, y_score_temp)
                ap = average_precision_score(true_labels_temp, y_score_temp)

                # write2excel(excel_filename_roc, pd.DataFrame(fpr, columns=['{}'.format('fpr')]), 'roc_{}_{}'.format(tar_dataset_name, label))
                # write2excel(excel_filename_roc, pd.DataFrame(tpr, columns=['{}'.format('tpr')]), 'roc_{}_{}'.format(tar_dataset_name, label))
                # write2excel(excel_filename_pr, pd.DataFrame(precision, columns=['{}'.format('precision')]), 'pr_{}_{}'.format(tar_dataset_name, label))
                # write2excel(excel_filename_pr, pd.DataFrame(recall, columns=['{}'.format('recall')]), 'pr_{}_{}'.format(tar_dataset_name, label))

                roc_auc = auc(fpr, tpr) ### 计算auc的值

                label_name = label_names[label]
                if label==0:
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.plot(fpr, tpr, color=color_list[label], lw=2, label='Unknown-{} (AUC={:.2f})'.format(label_name, roc_auc)) ###假正率为横坐标，真正率为纵坐标做曲线
                # plt.plot(recall, precision, color=color_list[label], lw=2, label='Unknown-{} (AP={:.2f})'.format(label_name, ap)) ###假正率为横坐标，真正率为纵坐标做曲线
                
                plt.xlim([0, 1.05])
                plt.ylim([0, 1.05])

                plt.legend(loc="lower right")
                plt.grid(True)

        plt.savefig(r'{}/ROC_{}_experiment_time({}).png'.format(args.img_path, case, experiment_time))
        # plt.show()
        plt.close()