import torch
import torch.nn.functional as F
from torch import nn

from config import args

if torch.cuda.is_available() and args.use_cuda:
    torch.cuda.manual_seed_all(args.seed)#为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    torch.manual_seed(args.seed)

class Net_5_2_base(nn.Module):
    def __init__(self):
        super(Net_5_2_base, self).__init__()
        # self.bn_1 = nn.BatchNorm1d(1)
        self.conv_1 = nn.Conv1d(1, 32, 64, 4)
        self.bn_2 = nn.BatchNorm1d(32)
        self.conv_2 = nn.Conv1d(32, 64, 3, 1)
        self.bn_3 = nn.BatchNorm1d(64)
        self.conv_3 = nn.Conv1d(64, 96, 3, 1)
        self.bn_4 = nn.BatchNorm1d(96)
        self.conv_4 = nn.Conv1d(96, 128, 3, 1)
        # self.bn_5 = nn.BatchNorm1d(128)
        # self.conv_5 = nn.Conv1d(128, 192, 3, 1)
        # self.bn_6 = nn.BatchNorm1d(192)
        # self.conv_6 = nn.Conv1d(192, 256, 3, 1)
        # self.bn_7 = nn.BatchNorm1d(256)
        # self.conv_7 = nn.Conv1d(256, 320, 3, 1)
        self.dropout_1 = nn.Dropout(0.25)
        self.dropout_2 = nn.Dropout(0.25)
        self.dropout_3 = nn.Dropout(0.25)
        self.dropout_4 = nn.Dropout(0.25)
        # self.dropout_5 = nn.Dropout(0.25)
        # self.dropout_6 = nn.Dropout(0.25)
        # self.dropout_7 = nn.Dropout(0.25)

    def forward(self, x):
        # if args.experiment_time>1:
        # x = self.bn_1(x)
        x = self.conv_1(x)
        x = F.relu(x)
        # x = self.dropout_1(x)
        x = F.max_pool1d(x, 2)

        # if args.experiment_time>2:
        x = self.bn_2(x)
        x = self.conv_2(x)
        x = F.relu(x)
        # x = self.dropout_2(x)
        x = F.max_pool1d(x, 2)

        # if args.experiment_time>3:
        x = self.bn_3(x)
        x = self.conv_3(x)
        x = F.relu(x)
        # x = self.dropout_3(x)
        x = F.max_pool1d(x, 2)

        # if args.experiment_time>4:
        x = self.bn_4(x)
        x = self.conv_4(x)
        x = F.relu(x)
        # x = self.dropout_4(x)
        x = F.max_pool1d(x, 2)

        # # if args.experiment_time>5:
        # x = self.bn_5(x)
        # x = self.conv_5(x)
        # x = F.relu(x)
        # # x = self.dropout_5(x)
        # x = F.max_pool1d(x, 2)

        # # if args.experiment_time>6:
        # x = self.bn_6(x)
        # x = self.conv_6(x)
        # x = F.relu(x)
        # # x = self.dropout_6(x)
        # x = F.max_pool1d(x, 2)

        # # if args.experiment_time>7:
        # x = self.bn_7(x)
        # x = self.conv_7(x)
        # x = F.relu(x)
        # # x = self.dropout_7(x)
        # x = F.max_pool1d(x, 2)    

        x = torch.flatten(x, 1)
        # print(output.size())

        return x


class Net_3_2_base(nn.Module):
    def __init__(self):
        super(Net_3_2_base, self).__init__()
        self.conv_1 = nn.Conv1d(1, 128, 64, 8)
        self.bn_1 = nn.BatchNorm1d(128)
        self.conv_2 = nn.Conv1d(128, 256, 5, 2)
        self.bn_2 = nn.BatchNorm1d(256)
        self.conv_3 = nn.Conv1d(256, 256, 3, 1)
        self.dropout_1 = nn.Dropout(0)

    def forward(self, input):
        x = self.conv_1(input)
        x = F.relu(x)
        x = F.max_pool1d(x, 4)

        x = self.bn_1(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        x = self.bn_2(x)
        x = self.conv_3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        output = torch.flatten(x, 1)
        # output = self.dropout_1(x)
        # print(output.size()

        return output


class Net_2_2_base(nn.Module):
    def __init__(self):
        super(Net_2_2_base, self).__init__()
        self.bn_1 = nn.BatchNorm1d(1)
        self.conv_1 = nn.Conv1d(1, 128, 256, 8)

        self.bn_2 = nn.BatchNorm1d(128)
        self.conv_2 = nn.Conv1d(128, 256, 5, 2)
        self.dropout_2 = nn.Dropout(0.25)

    def forward(self, x):
        # x = self.bn_1(x)
        x = self.conv_1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 4)

        x = self.bn_2(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        x = torch.flatten(x, 1)
        output = self.dropout_2(x)
        # print(output.size())

        return output


class Net_5_2(nn.Module):
    def __init__(self, class_num):
        # args.fc_5_2 = [2048, 7936, 7872, 5760, 3712, 2496, 1280, 320][args.experiment_time-1]
        super(Net_5_2, self).__init__()
        self.sharedNet = Net_5_2_base()

        self.bn_1_1 = nn.BatchNorm1d(args.fc_5_2)
        self.fc_1_1 = nn.Linear(args.fc_5_2, 512)
        self.dropout_1_1 = nn.Dropout(0.5)

        self.bn_1_2 = nn.BatchNorm1d(512)
        self.fc_1_2 = nn.Linear(512, class_num)
        self.dropout_1_2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.sharedNet(x)

        x = self.bn_1_1(x)
        x = self.fc_1_1(x)
        x = F.relu(x)
        # x = self.dropout_1_1(x)

        x = self.bn_1_2(x)
        x = self.fc_1_2(x)
        # x = self.dropout_1_2(x)

        return x


class Net_3_2(nn.Module):
    def __init__(self, class_num):
        super(Net_3_2, self).__init__()
        self.sharedNet = Net_3_2_base()

        self.bn_1_1 = nn.BatchNorm1d(1536)
        self.fc_1_1 = nn.Linear(1536, 512)
        self.dropout_1_1 = nn.Dropout(0.5)

        self.bn_1_2 = nn.BatchNorm1d(512)
        self.fc_1_2 = nn.Linear(512, class_num)
        self.dropout_1_2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.sharedNet(x)
        x = self.bn_1_1(x)
        x = self.fc_1_1(x)
        x = F.relu(x)
        x = self.dropout_1_1(x)
        x = self.bn_1_2(x)
        x = self.fc_1_2(x)
        output = self.dropout_1_2(x)

        return output


class Net_2_2(nn.Module):
    def __init__(self, class_num):
        super(Net_2_2, self).__init__()
        self.sharedNet = Net_2_2_base()

        self.bn_1_1 = nn.BatchNorm1d(args.fc_2_2) # 2816, 6912
        self.fc_1_1 = nn.Linear(args.fc_2_2, 512)
        self.dropout_1_1 = nn.Dropout(0.25)

        self.bn_1_2 = nn.BatchNorm1d(512)
        self.fc_1_2 = nn.Linear(512, class_num)
        self.dropout_1_2 = nn.Dropout(0.25)

    def forward(self, x):
        x = self.sharedNet(x)

        x = self.bn_1_1(x)
        x = self.fc_1_1(x)
        x = F.relu(x)
        x = self.dropout_1_1(x)

        x = self.bn_1_2(x)
        x = self.fc_1_2(x)
        # x = F.relu(x)
        output = self.dropout_1_2(x)

        return output

class OCNN(nn.Module):
    def __init__(self):
        super(OCNN, self).__init__()
        self.fc_1 = nn.Linear(42, 16)

        self.bn_2 = nn.BatchNorm1d(16)
        self.fc_2 = nn.Linear(16, 1)

    def forward(self, input):
        x = self.fc_1(input)
        x = torch.tanh(x)

        x = self.bn_2(x)
        x = self.fc_2(x)
        # x = torch.sigmoid(x)

        return x


class AE_encode(nn.Module):
    def __init__(self, data_length, class_num):
        super(AE_encode, self).__init__()
        self.bn_1 = nn.BatchNorm1d(data_length)
        self.fc_1 = nn.Linear(data_length, 342)
        
        self.bn_2 = nn.BatchNorm1d(342)
        self.fc_2 = nn.Linear(342, 170)

        self.bn_3 = nn.BatchNorm1d(170)
        self.lstm_1 = nn.LSTM(170, 170)

    def forward(self, input):
        x = self.bn_1(input)
        x = self.fc_1(input)
        x = F.relu(x)

        x = self.bn_2(x)
        x = self.fc_2(x)
        x = F.relu(x)

        x = self.bn_3(x)
        x = x.repeat(86, 1, 1)
        x, _ = self.lstm_1(x)

        return x

class AE_decode(nn.Module):
    def __init__(self, data_length, class_num):
        super(AE_decode, self).__init__()
        self.bn_1 = nn.BatchNorm1d(32)
        self.lstm_1 = nn.LSTM(170, 170)

        self.bn_2 = nn.BatchNorm1d(170)
        self.fc_2 = nn.Linear(170, 342)

        self.bn_3 = nn.BatchNorm1d(342)
        self.fc_3 = nn.Linear(342, data_length)

    def forward(self, input):
        x = self.bn_1(input)
        # x = input
        x, _ = self.lstm_1(x)
        x = torch.mean(x, dim=0)

        x = self.bn_2(x)
        x = self.fc_2(x)
        x = F.relu(x)

        x = self.bn_3(x)
        x = self.fc_3(x)
        x = torch.sigmoid(x)

        return x

class AE_encode_li(nn.Module):
    # 李巍华论文
    def __init__(self, data_length, class_num):
        super(AE_encode_li, self).__init__()
        self.fc_1 = nn.Linear(data_length, int(data_length*8/3))
        self.fc_2 = nn.Linear(int(data_length*8/3), class_num)

    def forward(self, input):
        print(input.size())
        x = self.fc_1(input)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)

        return x

class AE_decode_li(nn.Module):
    # 李巍华论文
    def __init__(self, data_length, class_num):
        super(AE_decode_li, self).__init__()
        self.fc_1 = nn.Linear(class_num, int(data_length))

    def forward(self, input):
        x = self.fc_1(input)
        x = torch.sigmoid(x)

        return x


class AE_encode_conv(nn.Module):
    def __init__(self):
        super(AE_encode_conv, self).__init__()
        self.conv_1 = nn.Conv1d(1, 64, 64, 8)
        self.bn_2 = nn.BatchNorm1d(64)
        self.conv_2 = nn.Conv1d(64, 128, 7, 5)
        self.bn_3 = nn.BatchNorm1d(128)
        self.conv_3 = nn.Conv1d(128, 128, 5, 3)
        self.bn_4 = nn.BatchNorm1d(128)
        self.conv_4 = nn.Conv1d(128, 128, 3, 2)
        self.bn_5 = nn.BatchNorm1d(128)
        self.conv_5 = nn.Conv1d(128, 128, 3, 2)

    def forward(self, input):
        x = self.conv_1(input)
        x = F.relu(x)
        # print(x.size())

        x = self.bn_2(x)
        x = self.conv_2(x)
        x = F.relu(x)
        # print(x.size())

        x = self.bn_3(x)
        x = self.conv_3(x)
        x = F.relu(x)
        # print(x.size())

        x = self.bn_4(x)
        x = self.conv_4(x)
        x = F.relu(x)
        # print(x.size())

        x = self.bn_4(x)
        x = self.conv_4(x)
        x = F.relu(x)
        # x = torch.flatten(x, 1)
        # print(x.size())

        return x


class AE_decode_conv(nn.Module):
    def __init__(self):
        super(AE_decode_conv, self).__init__()
        self.bn_1 = nn.BatchNorm1d(128)
        self.conv_t_1 = nn.ConvTranspose1d(128, 128, 3, 2)
        self.bn_2 = nn.BatchNorm1d(128)
        self.conv_t_2 = nn.ConvTranspose1d(128, 128, 3, 2)
        self.bn_3 = nn.BatchNorm1d(128)
        self.conv_t_3 = nn.ConvTranspose1d(128, 128, 5, 3, output_padding=2)
        self.bn_4 = nn.BatchNorm1d(128)
        self.conv_t_4 = nn.ConvTranspose1d(128, 64, 7, 5, output_padding=2)
        self.bn_5 = nn.BatchNorm1d(64)
        self.conv_t_5 = nn.ConvTranspose1d(64, 1, 64, 8)

    def forward(self, input):
        x = self.bn_1(input)
        x = self.conv_t_1(x)
        x = F.relu(x)
        # print(x.size())

        x = self.bn_2(x)
        x = self.conv_t_2(x)
        x = F.relu(x)
        # print(x.size())

        x = self.bn_3(x)
        x = self.conv_t_3(x)
        x = F.relu(x)
        # print(x.size())

        x = self.bn_4(x)
        x = self.conv_t_4(x)
        x = F.relu(x)
        # print(x.size())

        x = self.bn_5(x)
        x = self.conv_t_5(x)
        x = torch.sigmoid(x)
        # print(x.size())

        return x


class AE(nn.Module):
    def __init__(self, mode, data_length, class_num):
        super(AE, self).__init__()
        self.mode = mode
        if self.mode=='time_shao':
            self.encode = AE_encode(data_length, class_num)
            self.decode = AE_decode(data_length, class_num)
        elif self.mode=='conv':
            self.encode = AE_encode_conv()
            self.decode = AE_decode_conv()
        elif self.mode=='feature_li':
            self.encode = AE_encode_li(data_length, class_num)
            self.decode = AE_decode_li(data_length, class_num)
            
    def forward(self, input):
        encode = self.encode(input)
        output = self.decode(encode)

        return encode if self.mode!='time_shao' else torch.mean(encode, dim=0), output


class Common_Net(nn.Module):
    def __init__(self, class_num, net_class):
        super(Common_Net, self).__init__()
        self.net_class = net_class
        if net_class == '2_2':
            self.model = Net_2_2(class_num=class_num)
        elif net_class == '3_2':
            self.model = Net_3_2(class_num=class_num)
        elif net_class == '5_2':
            self.model = Net_5_2(class_num=class_num)
    
    def forward(self, source, target=0):
        if 'dan' in self.net_class:
            return self.model(source, target)
        else:
            return self.model(source)