from sfpt_copy import get_folder
import shutil

data_file_names = [r'Bearing/raw/A', r'Bearing/raw/D', r'Bearing/raw/G', r'Bearing/raw/O', r'Bearing/raw/Q', r'Gearbox/raw/J', r'Rotating/raw/M']

remote_file_path = [r'CloudData/data/{}'.format(x) for x in data_file_names]
local_file_path = [r'../data/{}'.format(x) for x in data_file_names]


for remote_path, local_path in zip(remote_file_path, local_file_path):
    get_folder(remote_path, local_path)

shutil.copytree('../data/Bearing/raw/A', '../data/Bearing/raw/B')
shutil.copytree('../data/Bearing/raw/A', '../data/Bearing/raw/C')
shutil.copytree('../data/Bearing/raw/D', '../data/Bearing/raw/E')
shutil.copytree('../data/Bearing/raw/D', '../data/Bearing/raw/F')
shutil.copytree('../data/Bearing/raw/G', '../data/Bearing/raw/H')
shutil.copytree('../data/Bearing/raw/G', '../data/Bearing/raw/I')
shutil.copytree('../data/Bearing/raw/O', '../data/Bearing/raw/P')
shutil.copytree('../data/Gearbox/raw/J', '../data/Gearbox/raw/K')
shutil.copytree('../data/Gearbox/raw/J', '../data/Gearbox/raw/L')
shutil.copytree('../data/Gearbox/raw/J', '../data/Gearbox/raw/L_0')
shutil.copytree('../data/Bearing/raw/Q', '../data/Bearing/raw/R')
shutil.copytree('../data/Rotating/raw/M', '../data/Rotating/raw/N')
shutil.copytree('../data/Rotating/raw/M', '../data/Rotating/raw/N_0')

print('Update data completed!')
