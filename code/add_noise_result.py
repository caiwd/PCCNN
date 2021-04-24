import json
import pandas as pd
import time
import os
import numpy as np
from config import args
from write_excel import write2excel
from multiprocessing import Pool

def task(file_name_list, json_data_list, method_list, feature_list, acc_type):
    all_result = {}
    for indexs, tar_dataset_name in enumerate(args.tar_dataset_name_list):
        datas = {}
        for method_name in method_list:
            for feature_type in feature_list:
                acc_list_2 = []
                for index,_ in enumerate(file_name_list):
                    acc_list_2.append(json_data_list[index][feature_type][method_name][tar_dataset_name][acc_type])
                datas['{}_{}'.format(feature_type, method_name)] = np.array(acc_list_2)
                if indexs==0:
                    all_result['{}_{}'.format(feature_type, method_name)] = 1/len(args.tar_dataset_name_list) * np.array(acc_list_2)
                else:
                    all_result['{}_{}'.format(feature_type, method_name)] += 1/len(args.tar_dataset_name_list) * np.array(acc_list_2)

        write2excel(r'{}/不同加噪比下的实验结果_{}_{}.xlsx'.format(args.result_path, 
            'gene' if args.generalization else 'baseline', acc_type), pd.DataFrame(datas), '{}'.format(tar_dataset_name))

    write2excel(r'{}/不同加噪比下的实验结果_{}_{}.xlsx'.format(args.result_path, 
            'gene' if args.generalization else 'baseline', acc_type), pd.DataFrame(all_result), '{}'.format('Aver'))

if __name__=='__main__':
    result_json = {} # 存储合并后的实验结果
    file_name_list = ['result_DB({})_{}_{}.json'.format(args.awgn_db_list[x-1], 
                                'gene' if args.generalization else 'baseline',
                                'noise' if args.add_awgn else 'no_noise')
                                for x in args.experiment_time_list]

    method_list = args.anormal_methods_list
    feature_list = args.features_list

    json_data_list = []
    for index,file_name in enumerate(file_name_list):
        file_path = os.path.join(args.result_path, file_name)
        with open(file_path, 'r') as f:
            json_data = json.load(f)
            json_data_list.append(json_data)

    p = Pool(3)
    for acc_type in ['known', 'unknown', 'aver']:
        p.apply_async(task, args=(file_name_list, json_data_list, method_list, feature_list, acc_type, ))

    p.close()
    p.join()
