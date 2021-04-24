import numpy as np
import torch

import _pickle as pickle
from config import args

np.random.seed(args.seed)

if torch.cuda.is_available() and args.use_cuda:
    torch.cuda.manual_seed_all(args.seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    torch.manual_seed(args.seed)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, categories, normal_01, add_noise, train_dataset=False):  # 需要加载的数据的标签
        super(MyDataset, self).__init__()
        self.normal_01 = normal_01
        self.add_noise = add_noise
        self.dataset_name = dataset_name
        self.categories = categories
        fh = open(dataset_name, 'r')
        datas = []
        num = 0
        for line in fh:
            words = line.split()
            if int(words[1]) < categories:
                if args.class_num < len(args.label_names) or not train_dataset: # 更新前或测试数据
                    datas.append((words[0], int(words[1])))
                else: # 更新后的训练数据
                    if int(words[1]) < categories-1: # 若标签不是新增类别
                        datas.append((words[0], int(words[1])))
                    else:
                        if num < args.new_class_num:
                            datas.append((words[0], int(words[1])))
                            num += 1
        num_list = []
        for i in range(categories):
            num = 0
            for _, label in datas:
                if label==i:
                    num += 1
            num_list.append(num)
        # print('{}'.format('Train' if train_dataset else 'Test'), dataset_name, categories, num_list, len(datas))
        self.samples_num = [num_list[0], num_list[-1]]
    
        self.datas = datas

    def __getitem__(self, index):
        fn, label = self.datas[index]
        with open(fn, 'rb') as f:
            data = pickle.load(f)
            data = np.reshape(data, (1, -1))
            data = data.astype('float32')

        if self.add_noise:
            noise = np.random.normal(1, 0.1, data.shape)  # 添加噪声
            data = data * noise
        if self.normal_01:
            if np.max(data) != np.min(data):
                data = (data-np.min(data, axis=1)) / (np.max(data, axis=1)-np.min(data, axis=1))  # 最大最小归一化
            else:
                pass
        data = torch.tensor(data).type(torch.FloatTensor)
        return data, int(label)

    def __len__(self):
        return len(self.datas)


def load_data(train_dataset_name, num_workers, batch_size=args.batch_size, data_type=args.data_type, categories=3, normal_01=True, add_noise=True):
    train_data = MyDataset(
            dataset_name=r'{}/label_{}_{}_train_same_num.meta'.format(args.label_path, data_type, train_dataset_name), categories=categories, normal_01=normal_01, add_noise=add_noise, train_dataset=True)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    test_data = MyDataset(
        dataset_name=r'{}/label_{}_{}_test_same_num.meta'.format(args.label_path, data_type, train_dataset_name), categories=categories+1, normal_01=normal_01, add_noise=False)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    return train_loader, test_loader, train_data.samples_num



if __name__ == '__main__':
    from matplotlib import pyplot as plt
    train_dataset_name = 'A'
    train_data_loader, test_data_loader, _ = load_data(
        train_dataset_name, categories=3, num_workers=args.num_workers)

    for datas, labels in train_data_loader:
        print(datas.shape, labels)
        plt.plot(datas[0][0])
        plt.show()
