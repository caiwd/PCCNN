import json
import pandas as pd
import os
import copy
import numpy as np
from config import args

def json_rate(json_data):
    # 每个实验结果转为列表存储
    for feature in json_data.keys(): # feature
        data_feature = json_data[feature]
        for method in data_feature.keys(): # method
            data_method = data_feature[method]
            for tar_dataset_name in data_method.keys(): # dataset_name
                data_dataset = data_method[tar_dataset_name]
                for category in data_dataset.keys(): # categories:
                     json_data[feature][method][tar_dataset_name][category]= [json_data[feature][method][tar_dataset_name][category]]
    return json_data

def combine_json(json_data, store_json_data):
    # 将多个json文件的相同键的列表内元素合并
    for feature in json_data.keys(): # feature
        data_feature = json_data[feature]
        for method in data_feature.keys(): # method
            data_method = data_feature[method]
            for tar_dataset_name in data_method.keys(): # dataset_name
                data_dataset = data_method[tar_dataset_name]
                for category in data_dataset.keys(): # categories:
                    one_value = json_data[feature][method][tar_dataset_name][category]
                    store_json_data[feature][method][tar_dataset_name][category].extend(one_value)
    return store_json_data

def calc_mean_std(datas, mean_std='mean'):
    # 计算每一种结果的均值或方差
    json_data = copy.deepcopy(datas)
    for feature in json_data.keys(): # feature
        data_feature = json_data[feature]
        for method in data_feature.keys(): # method
            data_method = data_feature[method]
            for tar_dataset_name in data_method.keys(): # dataset_name
                data_dataset = data_method[tar_dataset_name]
                for category in data_dataset.keys(): # categories:
                    if mean_std=='mean':
                        one_value = np.mean(np.array(json_data[feature][method][tar_dataset_name][category]))
                    elif mean_std=='std':
                        one_value = np.std(np.array(json_data[feature][method][tar_dataset_name][category]))
                    json_data[feature][method][tar_dataset_name][category]= round(one_value, 2)
    return json_data


def one_cloumn_list(result, category, tar_dataset_name, feature_list, method_list, std=True):
    value_list = []
    for index, feature in enumerate(feature_list):
        if std:
            temp = [str(round(result[0][feature][method][tar_dataset_name][category], 2))+'±'+str(round(result[1][feature][method][tar_dataset_name][category], 2)) for method in method_list]
        else:
            temp = [round(result[0][feature][method][tar_dataset_name][category], 2) for method in method_list]
        if index==0:
            value_list = temp
        else:
            value_list.extend(temp)

    return value_list

def return_class_num(tar_dataset_name):
    if tar_dataset_name in ['A', 'B', 'C']:
        class_num_list = [3, 4]
    elif tar_dataset_name in ['D', 'E', 'F']:
        class_num_list = [3, 4]
    elif tar_dataset_name in ['G', 'H', 'I']:
        class_num_list = [3, 4]
    elif tar_dataset_name in ['J', 'K', 'L', 'L_0']:
        class_num_list = [4, 5]
    elif tar_dataset_name in ['M', 'N', 'N_0']:
        class_num_list = [3, 4]
    elif tar_dataset_name in ['O', 'P', 'Q', 'R']:
        class_num_list = [2, 3]
    return class_num_list

if __name__=='__main__':
    result_json = {} # 存储合并后的实验结果
    file_name_list = ['result_experiment_time({})_{}_{}.json'.format(x, 
                                'gene' if args.generalization else 'baseline',
                                'noise' if args.add_awgn else 'no_noise')
                                for x in range(1,11)]
    method_list = args.anormal_methods_list
    feature_list = args.features_list
    for index,file_name in enumerate(file_name_list):
        file_path = os.path.join(args.result_path, file_name)
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        
        json_data = json_rate(json_data)

        if index==0:
            result_json = json_data
        else:
            result_json = combine_json(json_data, result_json)

    result = []
    for mean_std in ['mean', 'std']:
        json_result = calc_mean_std(result_json, mean_std=mean_std)
        with open(r'{}/result_{}_{}_{}.json'.format(args.result_path, mean_std, 
                                'gene' if args.generalization else 'baseline',
                                'noise' if args.add_awgn else 'no_noise'), 'w') as f:
            json.dump(json_result, f)
        result.append(json_result)

    for index in range(1):
        dict_datas = {}
        aver_result = {}
        for indexs, tar_dataset_name in enumerate(args.tar_dataset_name_list):
            class_num_list = return_class_num(tar_dataset_name)
            class_num = class_num_list[index]
            dict_datas['{}_known'.format(tar_dataset_name)] = one_cloumn_list(result, category='known', tar_dataset_name=tar_dataset_name, feature_list=feature_list, method_list=method_list)
            dict_datas['{}_unknown'.format(tar_dataset_name)] = one_cloumn_list(result, category='unknown', tar_dataset_name=tar_dataset_name, feature_list=feature_list, method_list=method_list)
            dict_datas['{}_aver'.format(tar_dataset_name)] = one_cloumn_list(result, category='aver', tar_dataset_name=tar_dataset_name, feature_list=feature_list, method_list=method_list)
        
        df = pd.DataFrame(dict_datas, index=['{}_{}'.format(feature, method) for feature in feature_list for method in method_list])

        df.to_excel(r'{}/更新前实验结果_含标准差_{}_{}.xlsx'.format(args.result_path,
                                'gene' if args.generalization else 'baseline',
                                'noise' if args.add_awgn else 'no_noise'))

    for index in range(1):
        dict_datas = {}
        for indexs, tar_dataset_name in enumerate(args.tar_dataset_name_list):
            class_num_list = return_class_num(tar_dataset_name)
            class_num = class_num_list[index]
            dict_datas['{}_known'.format(tar_dataset_name)] = one_cloumn_list(result, category='known', tar_dataset_name=tar_dataset_name, feature_list=feature_list, method_list=method_list, std=False)
            dict_datas['{}_unknown'.format(tar_dataset_name)] = one_cloumn_list(result, category='unknown', tar_dataset_name=tar_dataset_name, feature_list=feature_list, method_list=method_list, std=False)
            dict_datas['{}_aver'.format(tar_dataset_name)] = one_cloumn_list(result, category='aver', tar_dataset_name=tar_dataset_name, feature_list=feature_list, method_list=method_list, std=False)
        
            if indexs==0:
                dict_datas['Aver_known'] = np.array(dict_datas['{}_known'.format(tar_dataset_name)]) * 1/(len(args.tar_dataset_name_list))
                dict_datas['Aver_unknown'] = np.array(dict_datas['{}_unknown'.format(tar_dataset_name)]) * 1/(len(args.tar_dataset_name_list))
                dict_datas['Aver_aver'] = np.array(dict_datas['{}_aver'.format(tar_dataset_name)]) * 1/(len(args.tar_dataset_name_list))
            else:
                dict_datas['Aver_known'] += np.array(dict_datas['{}_known'.format(tar_dataset_name)]) * 1/(len(args.tar_dataset_name_list))
                dict_datas['Aver_unknown'] += np.array(dict_datas['{}_unknown'.format(tar_dataset_name)]) * 1/(len(args.tar_dataset_name_list))
                dict_datas['Aver_aver'] += np.array(dict_datas['{}_aver'.format(tar_dataset_name)]) * 1/(len(args.tar_dataset_name_list))

        df = pd.DataFrame(dict_datas, index=['{}_{}'.format(feature, method) for feature in feature_list for method in method_list])

        df.to_excel(r'{}/更新前实验结果_不含标准差_{}_{}.xlsx'.format(args.result_path,
                                'gene' if args.generalization else 'baseline',
                                'noise' if args.add_awgn else 'no_noise'), sheet_name='ALL')
