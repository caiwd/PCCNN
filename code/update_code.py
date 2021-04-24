import os
import socket

os.system('pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple')

import torch
if not torch.cuda.is_available():
    os.system('pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html')

from sfpt_copy import get, put

if socket.gethostname() == 'DESKTOP-E9LG3Q5':
    code_file_list = [x for x in os.listdir() if os.path.isfile(x)]
    remote_file_names = [r'CloudData/code/{}'.format(x) for x in code_file_list]
    put(code_file_list, remote_file_names)
else:
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("pan.blockelite.cn", port=20222, username="15501129681", password="nP7wHFCx")  # 连接SSH服务端
    transport = client.get_transport()
    sftp = paramiko.SFTPClient.from_transport(transport)
    code_file_list = [x for x in sftp.listdir(r'CloudData/code') if '.py' in x or '.txt' in x]
    remote_file_names = [r'CloudData/code/{}'.format(x) for x in code_file_list]
    sftp.close()

    get(remote_file_names, code_file_list)
