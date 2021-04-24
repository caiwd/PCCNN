import os
import progressbar
import time
 
def put(local_file_names, remote_file_names, is_print=True):
    import paramiko
    client = paramiko.SSHClient()   # 获取SSHClient实例
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("pan.blockelite.cn", port=20222, username="15501129681", password="nP7wHFCx")  # 连接SSH服务端
    transport = client.get_transport()   # 获取Transport实例
    
    # 创建sftp对象，SFTPClient是定义怎么传输文件、怎么交互文件
    sftp = paramiko.SFTPClient.from_transport(transport)
    
    # 将本地 api.py 上传至服务器 /www/test.py。文件上传并重命名为test.py
    # sftp.put("E:/test/api.py", "/www/test.py")
    for remote_file_name, local_file_name in zip(remote_file_names, local_file_names):
        sftp.put(local_file_name, remote_file_name)
    if is_print:
        print('{} - 数据上传成功！'.format(time.strftime('%Y-%m-%d_%H-%M-%S')))

    # 关闭连接
    client.close()

def get(remote_file_names, local_file_names):
    import paramiko
    client = paramiko.SSHClient()   # 获取SSHClient实例
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("pan.blockelite.cn", port=20222, username="15501129681", password="nP7wHFCx")  # 连接SSH服务端
    transport = client.get_transport()   # 获取Transport实例
    
    # 创建sftp对象，SFTPClient是定义怎么传输文件、怎么交互文件
    sftp = paramiko.SFTPClient.from_transport(transport)
    
    # 将服务器 /www/test.py 下载到本地 aaa.py。文件下载并重命名为aaa.py
    for remote_file_name, local_file_name in zip(remote_file_names, local_file_names):
        sftp.get(remote_file_name, local_file_name)
    print('{} - 数据下载成功！'.format(time.strftime('%Y-%m-%d_%H-%M-%S')))
    # 关闭连接
    client.close()

def get_folder(remote_path, local_path):
    import paramiko
    client = paramiko.SSHClient()   # 获取SSHClient实例
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect("pan.blockelite.cn", port=20222, username="15501129681", password="nP7wHFCx")  # 连接SSH服务端
    transport = client.get_transport()   # 获取Transport实例
    
    # 创建sftp对象，SFTPClient是定义怎么传输文件、怎么交互文件
    sftp = paramiko.SFTPClient.from_transport(transport)

    pro_ins = progressbar.ProgressBar()
    for file_name in pro_ins(sftp.listdir(remote_path)):
        if '.' in os.path.split(file_name)[-1]: # 若为文件
            if not os.path.exists(local_path):
                os.makedirs(local_path)
            sftp.get(r'{}/{}'.format(remote_path, file_name), r'{}/{}'.format(local_path, file_name))
        else: # 若为文件夹
            if not os.path.exists(r'{}/{}'.format(local_path, file_name)):
                os.makedirs(r'{}/{}'.format(local_path, file_name))
            for file_name_temp in sftp.listdir(r'{}/{}'.format(remote_path, file_name)):
                sftp.get(r'{}/{}/{}'.format(remote_path, file_name, file_name_temp), r'{}/{}/{}'.format(local_path, file_name, file_name_temp))

if __name__=='__main__':
    remote_file_names = [r'CloudData/code/config.py']
    local_file_names = [r'../data/config.py']
    get(remote_file_names, local_file_names) 