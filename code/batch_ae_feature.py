import os
from multiprocessing import Pool
import logging
import progressbar
from config import args
import time
from gpu_info import get_gpu_info

def ae_task_awgn(case, tar_dataset_name, data_type, class_num, experiment_time):
    # 得到时域数据的特征cnn
    if args.generalization:
        os.system('python feature_ae.py --case {} --tar_dataset_name {} --data_type {} --class_num {} --experiment_time {} --use_cuda --generalization --add_awgn'.format(case, tar_dataset_name, data_type, class_num, experiment_time))
    else:
        os.system('python feature_ae.py --case {} --tar_dataset_name {} --data_type {} --class_num {} --experiment_time {} --use_cuda --add_awgn'.format(case, tar_dataset_name, data_type, class_num, experiment_time))

def ae_task(case, tar_dataset_name, data_type, class_num, experiment_time):
    # 得到时域数据的特征cnn
    if args.generalization:
        os.system('python feature_ae.py --case {} --tar_dataset_name {} --data_type {} --class_num {} --experiment_time {} --use_cuda --generalization'.format(case, tar_dataset_name, data_type, class_num, experiment_time))
    else:
        os.system('python feature_ae.py --case {} --tar_dataset_name {} --data_type {} --class_num {} --experiment_time {} --use_cuda'.format(case, tar_dataset_name, data_type, class_num, experiment_time))

if __name__ == "__main__":
    experiment_time = args.experiment_time
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    # cnn特征
    add_awgn = args.add_awgn
    args.process_num = 1
    p = Pool(min(len(args.tar_dataset_name_list), args.process_num))
    pro_bar = progressbar.ProgressBar()
    # for tar_dataset_name in pro_bar(args.tar_dataset_name_list):
    for tar_dataset_name in args.tar_dataset_name_list:
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
        for class_num in [class_num_list[0]]:
            if args.add_awgn:
                p.apply_async(ae_task_awgn, args=(case, tar_dataset_name, 'ae_feature_{}'.format(class_num), class_num, experiment_time,))
            else:
                p.apply_async(ae_task, args=(case, tar_dataset_name, 'ae_feature_{}'.format(class_num), class_num, experiment_time,))
            time.sleep(10)
            memory_used_rate, cuda_used_rate = get_gpu_info()
            if memory_used_rate>0.75 or cuda_used_rate>0.85:
                while True:
                    memory_used_rate, cuda_used_rate = get_gpu_info()
                    if memory_used_rate<0.75 and cuda_used_rate<0.85:
                        break
                    time.sleep(3)
    p.close()
    p.join()

