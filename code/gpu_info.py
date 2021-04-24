import os
import numpy as np
import torch

def get_gpu_info():
    memory_used_rate_list, cuda_used_rate_list = [], []
    for i in range(5):
        if torch.cuda.is_available():
            nvidia_smi = os.popen('nvidia-smi')

            nvidia_smi_info = nvidia_smi.read()
            useful_info = [x for x in nvidia_smi_info.splitlines() if 'MiB' in x][0]
            used_memory, all_memory = [int(x[:-3]) for x in useful_info.split() if 'MiB' in x]
            cuda_used_rate_temp = [int(x[:-1])/100.0 for x in useful_info.split() if '%' in x][-1]
            memory_used_rate_temp = used_memory / all_memory

            memory_used_rate_list.append(memory_used_rate_temp)
            cuda_used_rate_list.append(cuda_used_rate_temp)
        else:
            memory_used_rate_list.append(0)
            cuda_used_rate_list.append(0)

    memory_used_rate = np.mean(np.array(memory_used_rate_list))
    cuda_used_rate = np.mean(np.array(cuda_used_rate_list))

    return memory_used_rate, cuda_used_rate
