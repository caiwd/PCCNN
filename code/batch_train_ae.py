import os
from multiprocessing import Pool
from config import args
import logging
import time
from progressbar import *
from gpu_info import get_gpu_info


def task_auto_encoder_finetune_weight(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num):
    if args.generalization:
        os.system('python auto_encoder.py --tar_dataset_name {} --class_num {} --data_type {} --epochs {} --case {} --lr {} --experiment_time {} --data_length {} --new_class_num {} --use_cuda --log_save --generalization --finetune --weight_loss'.format(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num))
    else:
        os.system('python auto_encoder.py --tar_dataset_name {} --class_num {} --data_type {} --epochs {} --case {} --lr {} --experiment_time {} --data_length {} --new_class_num {} --use_cuda --log_save --finetune --weight_loss'.format(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num))

def task_auto_encoder_finetune(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num):
    if args.generalization:
        os.system('python auto_encoder.py --tar_dataset_name {} --class_num {} --data_type {} --epochs {} --case {} --lr {} --experiment_time {} --data_length {} --new_class_num {} --use_cuda --log_save --generalization --finetune'.format(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num))
    else:
        os.system('python auto_encoder.py --tar_dataset_name {} --class_num {} --data_type {} --epochs {} --case {} --lr {} --experiment_time {} --data_length {} --new_class_num {} --use_cuda --log_save --finetune'.format(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num))

def task_auto_encoder_weight(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num):
    if args.generalization:
        os.system('python auto_encoder.py --tar_dataset_name {} --class_num {} --data_type {} --epochs {} --case {} --lr {} --experiment_time {} --data_length {} --new_class_num {} --use_cuda --log_save --generalization --weight_loss'.format(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num))
    else:
        os.system('python auto_encoder.py --tar_dataset_name {} --class_num {} --data_type {} --epochs {} --case {} --lr {} --experiment_time {} --data_length {} --new_class_num {} --use_cuda --log_save --weight_loss'.format(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num))


def task_auto_encoder(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num):
    if args.generalization:
        os.system('python auto_encoder.py --tar_dataset_name {} --class_num {} --data_type {} --epochs {} --case {} --lr {} --experiment_time {} --data_length {} --new_class_num {} --use_cuda --log_save --generalization'.format(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num))
    else:
        os.system('python auto_encoder.py --tar_dataset_name {} --class_num {} --data_type {} --epochs {} --case {} --lr {} --experiment_time {} --data_length {} --new_class_num {} --use_cuda --log_save'.format(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num))

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    # logging.warning('训练auto_encoder模型...')
    data_type = 'fre'
    lr = args.ae_train_lr
    experiment_time = args.experiment_time
    if args.finetune_weight:
        args.unknown_new_list = [1]
    else:
        args.unknown_new_list = [0]

    for index in args.unknown_new_list:
        args.process_num = 1
        p = Pool(min(len(args.tar_dataset_name_list), args.process_num))
        for new_class_num in args.new_class_num_list:
            if index==0:
                epochs = args.ae_train_epochs
            else:
                epochs = args.auto_encoder_update_epochs
            for tar_dataset_name in args.tar_dataset_name_list:
                if tar_dataset_name in ['J', 'K', 'L', 'L_0']:
                    case = 'Gearbox'
                    class_num_list = [4, 5]
                elif tar_dataset_name in ['M', 'N', 'N_0']:
                    case = 'Rotating'
                    class_num_list = [3, 4]
                else:
                    case = 'Bearing'
                    if tar_dataset_name in ['O', 'P', 'Q', 'R']:
                        class_num_list = [2, 3]
                    else:
                        class_num_list = [3, 4]
                class_num = class_num_list[index]
                data_length = args.fre_data_length
                if args.unknown_new_list == [0]:
                    p.apply_async(task_auto_encoder, args=(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num, ))
                elif args.finetune and args.weight_loss:
                    p.apply_async(task_auto_encoder_finetune_weight, args=(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num, ))
                elif args.finetune:
                    p.apply_async(task_auto_encoder_finetune, args=(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num, ))
                elif args.weight_loss:
                    p.apply_async(task_auto_encoder_weight, args=(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num, ))
                else:
                    p.apply_async(task_auto_encoder, args=(tar_dataset_name, class_num, data_type, epochs, case, lr, experiment_time, data_length, new_class_num, ))

                time.sleep(10)
                memory_used_rate, cuda_used_rate = get_gpu_info()
                if memory_used_rate>0.75 or cuda_used_rate>0.85:
                    while True:
                        memory_used_rate, cuda_used_rate = get_gpu_info()
                        if memory_used_rate<0.75 and cuda_used_rate<0.85:
                            break
                        time.sleep(3)
            if index==0:
                break
        p.close()
        p.join()
            