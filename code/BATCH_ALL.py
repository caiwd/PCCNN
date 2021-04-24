import os 
import logging
import time
from config import args
from multiprocessing import Pool


if __name__ =='__main__':
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    if args.tar_dataset_name_list:
        pass
    else:
        args.tar_dataset_name_list = args.default_tar_dataset_name_list

    # 分布式计算的相关设置
    current_work_index = range(args.rank, len(args.tar_dataset_name_list), args.world_size)
    current_tar_dataset_name_list = [args.tar_dataset_name_list[x] for x in current_work_index]

    tar_dataset_name_list = current_tar_dataset_name_list[0]
    for x in current_tar_dataset_name_list[1:]:
        tar_dataset_name_list += ' {}'.format(x)
    logging.warning('测试数据集共 {} 项，分别为：{}'.format(len(current_tar_dataset_name_list), tar_dataset_name_list))
    time.sleep(args.rank*45)

    logging.warning('安装相关软件包...')
    if os.path.exists('../../packages'):
        os.system('python install_packages.py')
    else:
        os.system('pip install -r requirements.txt')

    if args.task['use_gene_data']:
        gene_baseline = ' --generalization'
    else:
        gene_baseline = ''

    for experiment_time in args.experiment_time_list:
        # 数据分割，未加噪
        if (args.task['train_update_before_model'] or args.task['train_update_model'] or args.task['test_update_before_model'] or args.task['get_model_output']) and experiment_time<11:
            logging.warning('原始数据分割 - 未加噪数据 -【第 {} 次】...'.format(experiment_time))
            time.sleep(10)
            if args.task['train_update_before_model'] or args.task['test_update_before_model']:
                os.system('python batch_time_feature.py --experiment_time {} --tar_dataset_name_list {}{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))
            os.system('python batch_fre_feature.py --experiment_time {} --tar_dataset_name_list {}{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))
    

        # 训练更新前的模型
        if args.task['train_update_before_model'] and experiment_time<11:
            time.sleep(10)
            if 'cnn_feature' in args.features_list:
                logging.warning('训练卷积神经网络（CNN）- 训练更新前模型 - 【第 {} 次】...'.format(experiment_time))
                os.system('python batch_train_cnn.py --experiment_time {} --tar_dataset_name_list {}{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))
            if 'ae_feature' in args.features_list:
                logging.warning('训练自编码器（AE）- 训练更新前模型 - 【第 {} 次】...'.format(experiment_time))
                os.system('python batch_train_ae.py --experiment_time {} --tar_dataset_name_list {}{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))


        # 更新前模型的测试
        if args.task['test_update_before_model'] and experiment_time<11:
            time.sleep(10)
            if 'cnn_feature' in args.features_list:
                logging.warning('未加噪数据提取CNN特征 -【第 {} 次】...'.format(experiment_time))
                os.system('python batch_cnn_feature.py --experiment_time {} --tar_dataset_name_list {}{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))
            if 'ae_feature' in args.features_list:
                logging.warning('未加噪数据提取AE特征 -【第 {} 次】...'.format(experiment_time))
                os.system('python batch_ae_feature.py --experiment_time {} --tar_dataset_name_list {}{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))

            logging.warning('分类 - 未加噪测试 - 更新前模型 -【第 {} 次】...'.format(experiment_time))
            time.sleep(10)
            os.system('python batch_classify_method.py --experiment_time {} --classfy_times {} --tar_dataset_name_list {}{}'.format(experiment_time, 1, tar_dataset_name_list, gene_baseline))
        

        # 更新前模型的加噪测试
        if args.task['test_update_before_model_add_noise']:
            logging.warning('原始数据分割 - 原始数据加噪 -【DB = {}】...'.format(args.awgn_db_list[experiment_time-1]))
            time.sleep(10)
            os.system('python batch_time_feature.py --experiment_time {}  --tar_dataset_name_list {} --add_awgn{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))
            os.system('python batch_fre_feature.py --experiment_time {}  --tar_dataset_name_list {} --add_awgn{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))

            time.sleep(10)
            if 'cnn_feature' in args.features_list:
                logging.warning('提取CNN特征 - 提取加噪数据CNN特征 -【DB = {}】...'.format(args.awgn_db_list[experiment_time-1]))
                os.system('python batch_cnn_feature.py  --tar_dataset_name_list {} --add_awgn{}'.format(tar_dataset_name_list, gene_baseline))
            if 'ae_feature' in args.features_list:
                logging.warning('提取AE特征 - 提取加噪数据AE特征 -【DB = {}】...'.format(args.awgn_db_list[experiment_time-1]))
                os.system('python batch_ae_feature.py  --tar_dataset_name_list {} --add_awgn{}'.format(tar_dataset_name_list, gene_baseline))
        
            logging.warning('分类 - 加噪测试 - 更新前模型 -【DB = {}】...'.format(args.awgn_db_list[experiment_time-1]))
            time.sleep(10)
            os.system('python batch_classify_method.py --experiment_time {}  --tar_dataset_name_list {} --add_awgn{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))


        # 训练更新后的模型
        if args.task['train_update_model'] and experiment_time<11:        
            logging.warning('更新模型 - No_Weight_Loss + No_Fintune - 【第 {} 次】...'.format(experiment_time))
            time.sleep(10)
            os.system('python batch_train_cnn.py --experiment_time {} --tar_dataset_name_list {} --finetune_weight{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))

            logging.warning('更新模型 - Weight_Loss + Fintune -【第 {} 次】...'.format(experiment_time))
            time.sleep(10)
            os.system('python batch_train_cnn.py --experiment_time {} --tar_dataset_name_list {} --finetune_weight --finetune --weight_loss{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))

            logging.warning('更新模型 - Fintune - 【第 {} 次】...'.format(experiment_time))
            time.sleep(10)
            os.system('python batch_train_cnn.py --experiment_time {} --tar_dataset_name_list {} --finetune_weight --finetune{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))
        
            logging.warning('更新模型 - Weight_Loss - 【第 {} 次】...'.format(experiment_time))
            time.sleep(10)
            os.system('python batch_train_cnn.py --experiment_time {} --tar_dataset_name_list {} --finetune_weight --weight_loss{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))


        # 合并实验结果
        if args.task['combine_result']:
            # 未加噪
            os.system('python get_result.py --experiment_time {} --tar_dataset_name_list {}{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))
            # 加噪
            os.system('python get_result.py --experiment_time {}  --tar_dataset_name_list {} --add_awgn{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))
        
        # 获取CNN模型的输出
        if args.task['get_model_output'] and experiment_time<11:
            logging.warning('保存CNN模型的输出 -【{}】 ...'.format(experiment_time))
            os.system('python batch_get_model_result.py --experiment_time {} --tar_dataset_name_list {}{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))
        
        # 绘制ROC曲线
        if args.task['plot_roc'] and experiment_time<2:
            logging.warning('绘制ROC曲线 -【{}】 ...'.format(experiment_time))
            os.system('python plot_roc.py --experiment_time {} --tar_dataset_name_list {}{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))
        
        # 绘制混淆矩阵
        if args.task['plot_cm'] and experiment_time<2:
            logging.warning('绘制混淆矩阵 -【{}】 ...'.format(experiment_time))
            os.system('python plot_cm.py --experiment_time {} --tar_dataset_name_list {}{}'.format(experiment_time, tar_dataset_name_list, gene_baseline))

    # 合并实验结果到表格内
    if args.task['combine_result']:
        tasks = [
                'python update_before_result.py --tar_dataset_name_list {}{}'.format(tar_dataset_name_list, gene_baseline), 
                'python add_noise_result.py --add_awgn --tar_dataset_name_list {}{}'.format(tar_dataset_name_list, gene_baseline),
                'python update_result.py --tar_dataset_name_list {} --finetune --weight_loss{}'.format(tar_dataset_name_list, gene_baseline),
                'python update_result.py --tar_dataset_name_list {} --finetune{}'.format(tar_dataset_name_list, gene_baseline),
                'python update_result.py --tar_dataset_name_list {} --weight_loss{}'.format(tar_dataset_name_list, gene_baseline),
                'python update_result.py --tar_dataset_name_list {}{}'.format(tar_dataset_name_list, gene_baseline)
                ]
        p = Pool(3)
        for task in tasks:
            p.apply_async(os.system, args=(task, ))
        p.close()
        p.join()

    # 依据实验结果绘图
    os.system('python plot_result.py --tar_dataset_name_list {}{}'.format(tar_dataset_name_list, gene_baseline))
