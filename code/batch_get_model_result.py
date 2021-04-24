import os
import progressbar
from config import args
from multiprocessing import Pool

def task_finetune(case, class_num, tar_dataset_name, experiment_time):
    if args.generalization:
        os.system('python get_model_result.py --case {} --class_num {} --tar_dataset_name {} --experiment_time {} --use_cuda --generalization --finetune'.format(case, class_num, tar_dataset_name, experiment_time))
    else:
        os.system('python get_model_result.py --case {} --class_num {} --tar_dataset_name {} --experiment_time {} --use_cuda --finetune'.format(case, class_num, tar_dataset_name, experiment_time))

def task(case, class_num, tar_dataset_name, experiment_time):
    if args.generalization:
        os.system('python get_model_result.py --case {} --class_num {} --tar_dataset_name {} --experiment_time {} --use_cuda --generalization'.format(case, class_num, tar_dataset_name, experiment_time))
    else:
        os.system('python get_model_result.py --case {} --class_num {} --tar_dataset_name {} --experiment_time {} --use_cuda'.format(case, class_num, tar_dataset_name, experiment_time))


if __name__=='__main__':
    experiment_time = args.experiment_time
    p = Pool(args.process_num)
    for index in range(1):
        pro_bar = progressbar.ProgressBar()
        # for tar_dataset_name in pro_bar(args.tar_dataset_name_list):
        for tar_dataset_name in args.tar_dataset_name_list:
            if tar_dataset_name in ['J', 'K', 'L', 'L_0']:
                case = 'Gearbox'
            elif tar_dataset_name in ['M', 'N', 'N_0']:
                case = 'Rotating'
            else:
                case = 'Bearing'

            if tar_dataset_name in ['O', 'P', 'Q', 'R']:
                class_num_list = [2, 3]
            elif tar_dataset_name in ['J', 'K', 'L', 'L_0']:
                class_num_list = [4, 5]
            else:
                class_num_list = [3, 4]
            class_num = class_num_list[index]
            if args.finetune:
                p.apply_async(task_finetune, args=(case, class_num, tar_dataset_name, experiment_time,))
            else:
                p.apply_async(task, args=(case, class_num, tar_dataset_name, experiment_time,))

    p.close()
    p.join()