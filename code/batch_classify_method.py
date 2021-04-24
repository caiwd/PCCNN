import os
from multiprocessing import Pool
from progressbar import *
from config import args

def task_gene_awgn(case, tar_dataset_name, anormal_method, data_type, epochs, class_num, experiment_time):
    # 加噪声，泛化
    os.system('python classify_method.py --case {} --tar_dataset_name {} --anormal_method {} --data_type {} --epochs {} --class_num {} --experiment_time {} --log_save --generalization --add_awgn'.format(case, tar_dataset_name, anormal_method, data_type, epochs, class_num, experiment_time))

def task_awgn(case, tar_dataset_name, anormal_method, data_type, epochs, class_num, experiment_time):
    # 加噪声
    os.system('python classify_method.py --case {} --tar_dataset_name {} --anormal_method {} --data_type {} --epochs {} --class_num {} --experiment_time {} --log_save --add_awgn'.format(case, tar_dataset_name, anormal_method, data_type, epochs, class_num, experiment_time))

def task_gene(case, tar_dataset_name, anormal_method, data_type, epochs, class_num, experiment_time):
    # 加泛化
    os.system('python classify_method.py --case {} --tar_dataset_name {} --anormal_method {} --data_type {} --epochs {} --class_num {} --experiment_time {} --log_save --generalization'.format(case, tar_dataset_name, anormal_method, data_type, epochs, class_num, experiment_time))
    
def task(case, tar_dataset_name, anormal_method, data_type, epochs, class_num, experiment_time):
    os.system('python classify_method.py --case {} --tar_dataset_name {} --anormal_method {} --data_type {} --epochs {} --class_num {} --experiment_time {} --log_save'.format(case, tar_dataset_name, anormal_method, data_type, epochs, class_num, experiment_time))

if __name__ == "__main__":
    anormal_methods = args.anormal_methods_list
    epochs = args.classfy_times
    experiment_time = args.experiment_time
    
    widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),'  ', ETA()]
    pro_ins = ProgressBar(widgets=widgets)
    # for tar_dataset_name in pro_ins(args.tar_dataset_name_list):
    for tar_dataset_name in args.tar_dataset_name_list:
        for index in range(1):
            if tar_dataset_name in ['J', 'K', 'L', 'L_0']:
                case = 'Gearbox'
                class_num_list = [4, 5]
            elif tar_dataset_name in ['O', 'P', 'Q', 'R']:
                case = 'Bearing'
                class_num_list = [2, 3]
            elif tar_dataset_name in ['M', 'N', 'N_0']:
                case = 'Rotating'
                class_num_list = [3, 4]
            else:
                case = 'Bearing'
                class_num_list = [3, 4]
            class_num = class_num_list[index]

            features_list_cnn = ['cnn_feature'+'_{}'.format(class_num)]
            features_list_ae = ['ae_feature'+'_{}'.format(class_num)]
            features_list = [x for x in args.features_list if 'cnn' not in x and 'ae' not in x]
            if 'cnn_feature' in args.features_list:
                features_list.extend(features_list_cnn)
            if 'ae_feature' in args.features_list:
                features_list.extend(features_list_ae)

            max_process_num = min(args.process_num, len(features_list)*len(anormal_methods))
            p = Pool(max_process_num)
            for data_type in features_list:
                for anormal_method in anormal_methods:
                    if args.add_awgn and args.generalization:
                        p.apply_async(task_gene_awgn, args=(case, tar_dataset_name, anormal_method, data_type, epochs, class_num, experiment_time, ))
                    elif args.generalization:
                        p.apply_async(task_gene, args=(case, tar_dataset_name, anormal_method, data_type, epochs, class_num, experiment_time, ))
                    elif args.add_awgn:
                        p.apply_async(task_awgn, args=(case, tar_dataset_name, anormal_method, data_type, epochs, class_num, experiment_time, ))
                    else:
                        p.apply_async(task, args=(case, tar_dataset_name, anormal_method, data_type, epochs, class_num, experiment_time, ))
            p.close()
            p.join()