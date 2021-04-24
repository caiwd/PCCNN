import pandas as pd
import numpy as np
import json
from config import args
import os
import shutil

def read_excel(method, data_type, tar_dataset_name, class_num, pccnn=False):
    file_name = r'{}/{}_{}_{}_{}({})_{}_{}.xlsx'.format(args.log_path, tar_dataset_name, data_type, method,
            'DB' if args.add_awgn else 'experiment_time', 
            args.awgn_db_list[args.experiment_time-1] if args.add_awgn else args.experiment_time,
            'gene' if args.generalization else 'baseline',
            'noise' if args.add_awgn else 'no_noise')

    if os.path.exists(file_name):
        df = pd.read_excel(file_name, sheet_name='{}'.format(tar_dataset_name))
        result_1 = []
        result = {}
        if not pccnn:
            for value in df.values:
                correct_rate_str_1 = value[1].split('--')
                correct_rate_1 = [float(x[:-1]) for x in correct_rate_str_1]
                result_1.append(correct_rate_1)
            result_1 = np.array(result_1)
        mean_result_1 = np.mean(result_1, axis=0)

        result['known'] = round(mean_result_1[0], 2)
        result['unknown'] = round(mean_result_1[1], 2)
        result['aver'] = round(mean_result_1[2], 2)
        for index in range(class_num):
            result['{}_class'.format(index)] = round(mean_result_1[index+3], 2)

        return result
    return None


if __name__=='__main__':
    big_result = {}
    for data_type in args.features_list:
        all_result = {}
        for method in args.anormal_methods_list:
            result = {}
            for tar_dataset_name in args.tar_dataset_name_list:
                if tar_dataset_name in ['A', 'B', 'C']:
                    class_num = 4
                elif tar_dataset_name in ['D', 'E', 'F']:
                    class_num = 4
                elif tar_dataset_name in ['G', 'H', 'I']:
                    class_num = 4
                elif tar_dataset_name in ['J', 'K', 'L', 'L_0']:
                    class_num = 5
                elif tar_dataset_name in ['M', 'N', 'N_0']:
                    class_num = 4
                elif tar_dataset_name in ['O', 'P']:
                    class_num = 3
                elif tar_dataset_name in ['Q', 'R']:
                    class_num = 3

                excel_return = read_excel(method, data_type, tar_dataset_name, class_num, pccnn=False)
                if excel_return:
                    result[tar_dataset_name] = excel_return
            if result:
                all_result[method] = result
        big_result[data_type] = all_result
    
    if big_result[data_type]!={}:
        result_file_name = r'{}/result_{}({})_{}_{}.json'.format(args.result_path, 
                'DB' if args.add_awgn else 'experiment_time', 
                args.awgn_db_list[args.experiment_time-1] if args.add_awgn else args.experiment_time,
                'gene' if args.generalization else 'baseline',
                'noise' if args.add_awgn else 'no_noise')

        with open(result_file_name, 'w') as f:
            f.write(json.dumps(big_result))
        
        if os.path.exists(args.backup_path):
            shutil.copyfile(result_file_name, r'{}/{}'.format( args.backup_path, result_file_name[3:]))