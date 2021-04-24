from config import args
import torch
import torch.nn.functional as F
from load_data import load_data

import numpy as np
from cnn import get_c_v
from Networks import Common_Net


def p_2_class(c_v, p, class_num):
    pre_label = []
    for p_item in p:
        first_label = np.where(p_item == max(p_item))[0][0]
        if p_item[first_label] >= c_v[first_label]:
            pre_label.append(first_label)
        else:
            pre_label.append(class_num)

    return np.array(pre_label)


def load_test_train_data(tar_dataset_name, class_num, model):
    train_loader, test_loader, _ = load_data(
        tar_dataset_name, categories=class_num, num_workers=args.num_workers)

    for index, (datas, labels) in enumerate(test_loader):
        if torch.cuda.is_available() and args.use_cuda:
            datas = datas.cuda()
        output = model(datas)
        p = F.softmax(output, dim=1)

        if torch.cuda.is_available() and args.use_cuda:
            p = p.cpu()
            labels = labels.cpu()
        p = p.detach().numpy()
        labels = labels.detach().numpy()

        if index != 0:
            all_p = np.concatenate((all_p, p), axis=0)
            all_labels = np.concatenate((all_labels, labels), axis=0)
        else:
            all_p = p
            all_labels = labels
    return all_p, all_labels, train_loader


if __name__ == '__main__':
    class_num = args.class_num
    tar_dataset_name = args.tar_dataset_name
    experiment_time = args.experiment_time

    model = Common_Net(class_num=args.class_num, net_class=args.net_class)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')

    model_dict = model.state_dict()
    trained_model = torch.load(r'{}/{}_cnn_model_class_num({})_experiment_time({})_{}{}.pkl'.format(
        args.model_path, args.tar_dataset_name, args.class_num, args.experiment_time,
        'gene' if args.generalization else 'baseline', 
        '' if args.class_num!=len(args.label_names) else r'_{}_{}'.format(
        'finetune' if args.finetune else 'no_finetune', 
        'weight_loss' if args.weight_loss else 'no_weight_loss')))
    trained_model_dict = {k: v for k,
                          v in trained_model.items() if k in model_dict}
    model_dict.update(trained_model_dict)
    model.load_state_dict(model_dict)
    if torch.cuda.is_available() and args.use_cuda:
        model.cuda()

    model.eval()
    p, true_labels, train_loader = load_test_train_data(
        tar_dataset_name, class_num, model)

    # 保存预测标签\真实标签\cnn的输出概率
    c_v = get_c_v(model, train_loader, class_num=class_num)  # 置信度向量
    pre_labels = p_2_class(c_v, p, class_num)

    np.savetxt(r'{}/cnn_pre_{}_class_num({})_experiment_time({})_{}.txt'.format(
        args.cnn_output_path, tar_dataset_name, class_num, args.experiment_time, 
        'gene' if args.generalization else 'noise' if args.add_awgn else 'baseline'), pre_labels)
    np.savetxt(r'{}/cnn_true_{}_class_num({})_experiment_time({})_{}.txt'.format(
        args.cnn_output_path, tar_dataset_name, class_num, args.experiment_time, 
        'gene' if args.generalization else 'noise' if args.add_awgn else 'baseline'), true_labels)
    np.savetxt(r'{}/cnn_output_{}_class_num({})_experiment_time({})_{}.txt'.format(
        args.cnn_output_path, tar_dataset_name, class_num, args.experiment_time, 
        'gene' if args.generalization else 'noise' if args.add_awgn else 'baseline'), p)
