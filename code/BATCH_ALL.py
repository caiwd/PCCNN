import os 
import logging
import time
from config import args

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

for experiment_time in args.experiment_time_list:
    if args.task['time_split']:
        logging.warning('Splitting time domain data, extracting features【{}】...'.format(experiment_time))
        time.sleep(10)
        os.system('python batch_time_feature.py --experiment_time {}'.format(experiment_time))

    if args.task['cnn_train']:
        logging.warning('Training PCCNN model 【{}】...'.format(experiment_time))
        time.sleep(10)
        os.system('python batch_train_pccnn.py --experiment_time {}'.format(experiment_time))

    if args.task['cnn_feature']:
        logging.warning('Extracting CNN features【{}】...'.format(experiment_time))
        time.sleep(10)
        os.system('python batch_cnn_feature.py --experiment_time {}'.format(experiment_time))

    if args.task['classify']:
        logging.warning('classify【{}】...'.format(experiment_time))
        time.sleep(10)
        os.system('python batch_classify_method.py --experiment_time {}'.format(experiment_time))

    if args.task['excel2json']:
        logging.warning('Combining experimental results【{}】...'.format(experiment_time))
        time.sleep(10)
        os.system('python get_result.py --experiment_time {}'.format(experiment_time))

if args.task['combine_json']:
    logging.warning('Storing experimental results in a table...')
    time.sleep(10)
    os.system('python combine_json.py')