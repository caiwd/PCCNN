import os
import time
from multiprocessing import Pool
from config import args
from progressbar import *

def fre_task_gene_awgn(case, tar_dataset_name, data_space, data_type, experiment_time):
    # 分割时域数据，泛化，加噪
    os.system('python split_data.py --case {} --tar_dataset_name {} --data_space {} --data_type {} --experiment_time {} --generalization --add_awgn'.format(case, tar_dataset_name, data_space, data_type, experiment_time))

def fre_task_gene(case, tar_dataset_name, data_space, data_type, experiment_time):
    # 分割时域数据，泛化
    os.system('python split_data.py --case {} --tar_dataset_name {} --data_space {} --data_type {} --experiment_time {} --generalization'.format(case, tar_dataset_name, data_space, data_type, experiment_time))

def fre_task_awgn(case, tar_dataset_name, data_space, data_type, experiment_time):
    # 分割时域数据，加噪
    os.system('python split_data.py --case {} --tar_dataset_name {} --data_space {} --data_type {} --experiment_time {} --add_awgn'.format(case, tar_dataset_name, data_space, data_type, experiment_time))

def fre_task(case, tar_dataset_name, data_space, data_type, experiment_time):
    # 分割时域数据
    os.system('python split_data.py --case {} --tar_dataset_name {} --data_space {} --data_type {} --experiment_time {}'.format(case, tar_dataset_name, data_space, data_type, experiment_time))

if __name__ == "__main__":
    data_type = 'fre'
    experiment_time = args.experiment_time
    p = Pool(min(args.process_num, len(args.tar_dataset_name_list)))

    widgets = ['Progress: ',Percentage(), ' ', Bar('#'),' ', Timer(),'  ', ETA()]
    pro_ins = ProgressBar(widgets=widgets)
    # for tar_dataset_name in pro_ins(args.tar_dataset_name_list):
    for tar_dataset_name in args.tar_dataset_name_list:
        if tar_dataset_name in ['A', 'B', 'C']:
            case = 'Bearing'
            space = args.split_spcae_cwru
        elif tar_dataset_name in ['D', 'E', 'F']:
            case = 'Bearing'
            space = args.split_spcae_rmsf
        elif tar_dataset_name in ['G', 'H', 'I']:
            case = 'Bearing'
            space = args.split_spcae_gb
        elif tar_dataset_name in ['J', 'K', 'L', 'L_0']:
            case = 'Gearbox'
            space = args.split_spcae_gb
        elif tar_dataset_name in ['O', 'P']:
            case = 'Bearing'
            space = args.split_spcae_mfpt
        elif tar_dataset_name in ['M', 'N', 'N_0']:
            case = 'Rotating'
            space = args.split_spcae_abvt
        elif tar_dataset_name in ['Q', 'R']:
            case = 'Bearing'
            space = args.split_spcae_canda
        if args.generalization and args.add_awgn:
            p.apply_async(fre_task_gene_awgn, args=(case, tar_dataset_name, space, data_type, experiment_time, ))
        elif args.generalization:
            p.apply_async(fre_task_gene, args=(case, tar_dataset_name, space, data_type, experiment_time, ))
        elif args.add_awgn:
            p.apply_async(fre_task_awgn, args=(case, tar_dataset_name, space, data_type, experiment_time, ))
        else:
            p.apply_async(fre_task, args=(case, tar_dataset_name, space, data_type, experiment_time, ))
    p.close()
    p.join()
