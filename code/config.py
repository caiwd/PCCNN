import argparse
import os
import time
import random

parser = argparse.ArgumentParser()

parser.add_argument('--rank', type=int,default=0, help='Rank of the current process.')
parser.add_argument('--world-size',type=int, default=1, help='Number of processes participating in the job.')
parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP of the current rank 0.')
parser.add_argument('--port', type=str, default='20000', help='Port of the current rank 0.')

parser.add_argument('--case', type=str, default='CWRU', help='试验台名称')
parser.add_argument('--src_dataset_name', type=str, default='A', help='原域数据集')
parser.add_argument('--tar_dataset_name', type=str, default='C', help='目标域数据集')
parser.add_argument('--batch_size', type=int, default=32, help='批样本大小')
parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.001, help='初始学习率')
parser.add_argument('--num_workers', type=int, default=0, help='Loader线程数')
parser.add_argument('--samples_num', type=int, default=1024, help='每类生成的样本数量')
parser.add_argument('--momentum', type=float, default=0.9, help='优化器参数')
parser.add_argument('--class_num', type=int, default=3, help='分类数')
parser.add_argument('--net_class', type=str, default='5_2', help='模型结构， 2_2, 5_2, dan_2_2, dan_5_2')
parser.add_argument('--log_save', action='store_true', help='是否保存日志')
parser.add_argument('--train_test', type=str, default='train', help='训练or测试')
parser.add_argument('--use_cuda', action='store_true', help='是否使用cuda加速')
parser.add_argument('--step_size', type=int, default=100, help='每间隔多少轮学习率减小为多少倍')
parser.add_argument('--label_path', type=str, default=r'../label', help='标签存放在的路径')
parser.add_argument('--model_path', type=str, default=r'../model', help='模型文件存放在的路径')
parser.add_argument('--log_path', type=str, default=r'../log', help='日志存放在的路径')
parser.add_argument('--data_path', type=str, default=r'../data', help='数据存放在的路径')
parser.add_argument('--result_path', type=str, default=r'../result', help='结果存放在的路径')
parser.add_argument('--img_path', type=str, default=r'../img', help='存放图片的路径')
parser.add_argument('--pccnn_output_path', type=str, default=r'../pccnn_output', help='PCCCNN输出文件的路径')
parser.add_argument('--cnn_output_path', type=str, default=r'../cnn_output', help='CNN输出文件的路径')
parser.add_argument('--backup_path', type=str, default=r'/public/home/caiwd/backup', help='备份文件路径')
parser.add_argument('--data_type', type=str, default='fre', help='数据类型, fre, time, feature')
parser.add_argument('--split_rate_list', type=list, default=[
                    [[[0, 0.5]], [[0.5, 1.0]]],
                    [[[0.1, 0.6]], [[0, 0.1], [0.6, 1.0]]], 
                    [[[0.2, 0.7]], [[0, 0.2], [0.7, 1.0]]], 
                    [[[0.3, 0.8]], [[0, 0.3], [0.8, 1.0]]], 
                    [[[0.4, 0.9]], [[0, 0.4], [0.9, 1.0]]], 
                    [[[0.5, 1.0]], [[0, 0.5]]], 
                    [[[0.6, 1.0], [0, 0.1]], [[0.1, 0.6]]], 
                    [[[0.7, 1.0], [0, 0.2]], [[0.2, 0.7]]], 
                    [[[0.8, 1.0], [0, 0.3]], [[0.3, 0.8]]], 
                    [[[0.9, 1.0], [0, 0.4]], [[0.4, 0.9]]]
                    ], help='训练集与测试集的划分比例')
parser.add_argument('--anormal_method', type=str, default='svdd', help='异常检测方法')
parser.add_argument('--threshold_std_times', type=float, default=10, help='异常检测阈值几倍的标准差')
parser.add_argument('--ocsvm_gamma', type=float, default=0.35, help='OCSVM参数gamma')
parser.add_argument('--ocsvm_nu', type=float, default=0.25, help='OCSVM参数nu')
parser.add_argument('--feature_name', type=str, default='ae_feature', help='pca降维的特征')
parser.add_argument('--experiment_time', type=int, default=1, help='实验重复的次数')
# parser.add_argument('--features_list', type=list, default=['feature', 'cnn_feature', 'ae_feature'], help='特征列表')
parser.add_argument('--features_list', type=list, default=['ae_feature'], help='特征列表')
parser.add_argument('--class_num_list', type=list, default=[3, 4], help='几分类问题')
parser.add_argument('--classfy_times', type=int, default=10, help='分类算法重复运行的次数')
parser.add_argument('--data_length', type=int, default=4096, help='原始数据长度')
parser.add_argument('--fre_data_length', type=int, default=2048, help='频域数据长度')
parser.add_argument('--data_space', type=int, default=512, help='切分数据时的数据间隔')
parser.add_argument('--split_spcae_cwru', type=int, default=64*2, help='CWRU数据间隔')
parser.add_argument('--split_spcae_rmsf', type=int, default=64*18, help='RMSF数据间隔')
parser.add_argument('--split_spcae_mfpt', type=int, default=64*2, help='MFPT数据间隔')
parser.add_argument('--split_spcae_gb', type=int, default=64*10, help='Gearbox数据间隔')
parser.add_argument('--split_spcae_abvt', type=int, default=64*8, help='12G数据数据间隔')
parser.add_argument('--split_spcae_canda', type=int, default=64*12, help='加拿大渥太华数据间隔')
parser.add_argument('--new_class_num', type=int, default=10, help='新类别样本数目')
parser.add_argument('--unknown_new_list', type=list, default=[0], help='0代表更新前，1代表更新后')
parser.add_argument('--generalization', action='store_true', help='是否测试泛化性能')
parser.add_argument('--add_awgn', action='store_true', help='是否添加高斯噪声')
parser.add_argument('--finetune', action='store_true', help='是否Finetune')
parser.add_argument('--weight_loss', action='store_true', help='是否添加标签权重')
parser.add_argument('--finetune_weight', action='store_true', help='是否验证finetune和标签权重')
parser.add_argument('--default_tar_dataset_name_list', type=list, default=['A','B','C','D','E','F','G','H','I','J','K','L','L_0','M','N','N_0','O','P','Q','R'], help='需要测试的数据集列表，默认所有')



###################################################################################################################################################################################################
# 修改以下部分，其余部分勿改
parser.add_argument('--anormal_methods_list', type=list, default=['gmm', 'ocsvm', 'lof', 'isofore', 'sp'], help='分类方法列表')
parser.add_argument('--cnn_train_lr', type=float, default=0.0001, help='CNN学习率')
parser.add_argument('--ae_train_lr', type=float, default=0.0001, help='auto_encoder学习率')
parser.add_argument('--lr_rate', type=float, default=1, help='fintune学习率系数')
parser.add_argument('--process_num', type=int, default=5, help='线程数')
parser.add_argument('--new_class_num_list', type=list, default=[int(1.4**x) for x in range(2, 25)], help='新类别样本数目')
parser.add_argument('--log_interval', type=int, default=50, help='CNN训练时测试的间隔(epoch)')
parser.add_argument('--awgn_db_list', type=list, default=[x*1.5 for x in range(0, 25)], help='测试时加噪DB')
parser.add_argument('--experiment_time_list', type=list, default=[x for x in range(1, 26)], help='实验重复的次数列表')
parser.add_argument('--cnn_train_epochs', type=int, default=30, help='CNN训练的次数')
parser.add_argument('--ae_train_epochs', type=int, default=50, help='AE训练的次数')
parser.add_argument('--cnn_update_epochs', type=int, default=30, help='CNN更新的次数')
parser.add_argument('--tar_dataset_name_list', nargs='+', help='需要测试的数据集列表')
parser.add_argument('--task', type=dict, default={'use_gene_data': True, 'train_update_before_model': False,
                        'test_update_before_model': False, 'train_update_model': False,
                        'test_update_before_model_add_noise': False, 'combine_result': True,
                        'get_model_output': False, 'plot_roc': False, 'plot_cm': False,
                        'plot_compared_method': False, 'plot_noise': False, 'plot_update_result': False},
                        help='需要进行的任务')
###################################################################################################################################################################################################



parser.add_argument('--seed', type=int, default=12345, help='随机种子')
args = parser.parse_args()
# 如果需要指定随机种子，注释下一行，并在上两行修改默认值
args.seed = random.randint(0, 65535)

parser.add_argument('--fc_2_2', type=int, default=3328, help='2_2网络全连接层输入')
parser.add_argument('--fc_5_2', type=int, default=3712, help='5_2网络全连接层输入0:2048, 1:7936, 2:7872, 3:5760, 4:3712, 5:2496')

if args.case=='Bearing' and args.tar_dataset_name in ['A', 'B', 'C']:
    parser.add_argument('--fs', type=int, default=12*1000, help='采样频率')
    parser.add_argument('--k', type=int, default=1000, help='k的取值')
    if args.tar_dataset_name=='A':
        parser.add_argument('--label_names', type=list, default=['Normal', 'OR', 'IR', 'BF'], help='已知类别标签')
    elif args.tar_dataset_name=='B':
        parser.add_argument('--label_names', type=list, default=['Normal', 'OR', 'BF', 'IR'], help='已知类别标签')
    elif args.tar_dataset_name=='C':
        parser.add_argument('--label_names', type=list, default=['Normal', 'IR', 'BF', 'OR'], help='已知类别标签')

elif args.case=='Bearing' and args.tar_dataset_name in ['D', 'E', 'F']:
    parser.add_argument('--fs', type=int, default=25*1024, help='采样频率')
    parser.add_argument('--k', type=int, default=1024, help='k的取值')
    if args.tar_dataset_name=='D':
        parser.add_argument('--label_names', type=list, default=['Normal', 'OR', 'IR', 'MF'], help='已知类别标签')
    elif args.tar_dataset_name=='E':
        parser.add_argument('--label_names', type=list, default=['Normal', 'OR', 'MF', 'IR'], help='已知类别标签')
    elif args.tar_dataset_name=='F':
        parser.add_argument('--label_names', type=list, default=['Normal', 'IR', 'MF', 'OR'], help='已知类别标签')

elif args.case=='Bearing' and args.tar_dataset_name in ['G', 'H', 'I']:
    parser.add_argument('--fs', type=int, default=2*1024, help='采样频率')
    parser.add_argument('--k', type=int, default=1000, help='k的取值')
    if args.tar_dataset_name=='G':
        parser.add_argument('--label_names', type=list, default=['health', 'outer', 'inner', 'ball'], help='已知类别标签')
    elif args.tar_dataset_name=='H':
        parser.add_argument('--label_names', type=list, default=['health', 'outer', 'ball', 'inner'], help='已知类别标签')
    elif args.tar_dataset_name=='I':
        parser.add_argument('--label_names', type=list, default=['health', 'inner', 'ball', 'outer'], help='已知类别标签')

elif args.case=='Gearbox' and args.tar_dataset_name in ['J', 'K', 'L', 'L_0']:
    parser.add_argument('--fs', type=int, default=2*1000, help='采样频率')
    parser.add_argument('--k', type=int, default=1000, help='k的取值')
    if args.tar_dataset_name=='J':
        parser.add_argument('--label_names', type=list, default=['Health', 'Miss', 'Root', 'Chipped', 'Surface'], help='已知类别标签')
    elif args.tar_dataset_name=='K':
        parser.add_argument('--label_names', type=list, default=['Health', 'Miss', 'Root', 'Surface', 'Chipped'], help='已知类别标签')
    elif args.tar_dataset_name=='L':
        parser.add_argument('--label_names', type=list, default=['Health', 'Miss', 'Chipped', 'Surface', 'Root'], help='已知类别标签')
    elif args.tar_dataset_name=='L_0':
        parser.add_argument('--label_names', type=list, default=['Health', 'Root', 'Chipped', 'Surface', 'Miss'], help='已知类别标签')

elif args.case=='Bearing' and args.tar_dataset_name in ['O', 'P']:
    parser.add_argument('--k', type=int, default=1000, help='k的取值')
    parser.add_argument('--fs', type=int, default=24414, help='采样频率')
    if args.tar_dataset_name=='O':
        parser.add_argument('--label_names', type=list, default=['Normal', 'OR', 'IR'], help='已知类别标签')
    elif args.tar_dataset_name=='P':
        parser.add_argument('--label_names', type=list, default=['Normal', 'IR', 'OR'], help='已知类别标签')

elif args.case=='Bearing' and args.tar_dataset_name in ['Q', 'R']:
    parser.add_argument('--k', type=int, default=1000, help='k的取值')
    parser.add_argument('--fs', type=int, default=25000, help='采样频率')
    if args.tar_dataset_name=='Q':
        parser.add_argument('--label_names', type=list, default=['Normal', 'OR', 'IR'], help='已知类别标签')
    elif args.tar_dataset_name=='R':
        parser.add_argument('--label_names', type=list, default=['Normal', 'IR', 'OR'], help='已知类别标签')

elif args.case=='Rotating' and args.tar_dataset_name in ['M', 'N', 'N_0']:
    parser.add_argument('--k', type=int, default=1000, help='k的取值')
    parser.add_argument('--fs', type=int, default=25000, help='采样频率')
    if args.tar_dataset_name=='M':
        parser.add_argument('--label_names', type=list, default=['Normal', 'MIS', 'UBA', 'ORF'], help='已知类别标签')
    elif args.tar_dataset_name=='N':
        parser.add_argument('--label_names', type=list, default=['Normal', 'MIS', 'ORF', 'UBA'], help='已知类别标签')
    elif args.tar_dataset_name=='N_0':
        parser.add_argument('--label_names', type=list, default=['Normal', 'ORF', 'UBA', 'MIS'], help='已知类别标签')

parser.add_argument('--log_name', type=str, default=r'../log/{}_proposed_{}.txt'.format(
    args.tar_dataset_name, time.strftime('%Y-%m-%d_%H-%M-%S')), help='')
args = parser.parse_args()

if not os.path.exists(args.label_path):
    os.makedirs(args.label_path)
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)
if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)
if not os.path.exists(args.img_path):
    os.makedirs(args.img_path)
if not os.path.exists(args.cnn_output_path):
    os.makedirs(args.cnn_output_path)
