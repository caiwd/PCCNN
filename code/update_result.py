import pandas as pd
import numpy as np
from config import args
import os
from write_excel import write2excel
from update_before_result import return_class_num

args.experiment_time_list = range(1, 11)

def process_data_to_excel(tar_dataset_name, class_num):
    
    all_result = []

    for experiment_time in args.experiment_time_list:
        file_name = r'{}/{}_cnn_experiment_time({})_{}_{}_{}.xlsx'.format(
        args.log_path, tar_dataset_name, experiment_time,
        'gene' if args.generalization else 'baseline',
        'finetune' if args.finetune else 'no_finetune',
        'weight_loss' if args.weight_loss else 'no_weight_loss')
        
        df = pd.read_excel(file_name, sheet_name='{}'.format(tar_dataset_name))
        result_data = []
        for values in df.values[-1:]:
            result_one_times = []
            for index, _ in enumerate(args.new_class_num_list):
                try:
                    correct_rate_str_temp = values[index+1].split('--')
                    correct_rate_temp = np.array([float(x[:-1]) for x in correct_rate_str_temp][-class_num:])
                    result_one_times.append( [np.mean(correct_rate_temp[:-1]), np.mean(correct_rate_temp[-1]), np.mean(correct_rate_temp)] )
                except:
                    pass
            result_data.append(result_one_times)
        result_data_one_experiment = np.mean(np.array(result_data), axis=0)
        all_result.append(result_data_one_experiment)

    min_len = min([x.shape[0] for x in all_result])
    all_result = [x[:min_len] for x in all_result]

    all_result = np.array(all_result)
    all_result_mean = np.mean(all_result, axis=0)
    datas = {}
    for index, acc_type in enumerate(['known', 'new', 'aver']):
        datas['{}'.format(acc_type)] = np.array(all_result_mean[:, index])

    write2excel(r'{}/不同新类别数目的实验结果_{}_{}_{}.xlsx'.format(args.result_path, 
            'gene' if args.generalization else 'baseline',
            'finetune' if args.finetune else 'no_finetune',
            'weight_loss' if args.weight_loss else 'no_weight_loss'), 
            pd.DataFrame(datas), '{}'.format(tar_dataset_name))


if __name__=='__main__':
    for index, tar_dataset_name in enumerate(args.tar_dataset_name_list):
        class_num = return_class_num(tar_dataset_name)[-1]
        process_data_to_excel(tar_dataset_name, class_num)

    all_result = {}
    aver_result = {}
    filepath = r'{}/不同新类别数目的实验结果_{}_{}_{}.xlsx'.format(args.result_path, 
            'gene' if args.generalization else 'baseline',
            'finetune' if args.finetune else 'no_finetune',
            'weight_loss' if args.weight_loss else 'no_weight_loss')

    for indexs, tar_dataset_name in enumerate(args.tar_dataset_name_list):
        df = pd.read_excel(filepath, sheet_name='{}'.format(tar_dataset_name))
        for index, acc_type in enumerate(['known', 'new', 'aver']):
            all_result['{}_{}'.format(tar_dataset_name, acc_type)] = df[acc_type]
            if indexs==0:
                aver_result['{}'.format(acc_type)] = df[acc_type] * 1/len(args.tar_dataset_name_list)
            else:
                aver_result['{}'.format(acc_type)] += df[acc_type] * 1/len(args.tar_dataset_name_list)

    write2excel(filepath, pd.DataFrame(all_result), 'ALL')

    for index, acc_type in enumerate(['known', 'new', 'aver']):
        write2excel(filepath, pd.DataFrame(aver_result['{}'.format(acc_type)]), 'Aver')
        

