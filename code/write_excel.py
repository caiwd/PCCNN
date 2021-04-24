import logging
import os
import random
import time

import pandas as pd


def write2excel_base(excel_path, datas, sheet):
    exist_datas = {}
    if not os.path.exists(excel_path):
        nan_excel = pd.DataFrame()
        nan_excel.to_excel(excel_path)
        exist_datas[sheet] = datas
    else:
        f = pd.ExcelFile(excel_path)
        for name in f.sheet_names:
            exist_datas[name] = pd.read_excel(excel_path, name)
        if sheet in f.sheet_names:  # 是否存在sheet
            exist_datas[sheet][datas.columns.item()] = datas
        else:
            exist_datas[sheet] = datas
    writer = pd.ExcelWriter(excel_path)
    for name in exist_datas:
        try:
            save_data = exist_datas[name].drop(labels='Unnamed: 0', axis=1)
        except:
            save_data = exist_datas[name]
        save_data.to_excel(writer, sheet_name=name)
    writer.save()
    for index in range(5):
        if os.path.exists(r'{}/{}_{}'.format(os.path.split(excel_path)[0], index, os.path.basename(excel_path))):
            try:
                os.remove(r'{}/{}_{}'.format(os.path.split(excel_path)[0], index, os.path.basename(excel_path)))
            except:
                pass


def write2excel(excel_path, datas, sheet):
    for index in range(5):
        if '{}_{}'.format(index, os.path.basename(excel_path)) in os.listdir(os.path.split(excel_path)[0]):
            time.sleep(random.randint(5, 11))
            continue
        with open(r'{}/{}_{}'.format(os.path.split(excel_path)[0], index, os.path.basename(excel_path)), 'w') as f:
            time.sleep(random.random())
    write2excel_base(excel_path, datas, sheet)
            
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    excel_path = '1.xlsx'
    sheet = '2'
    for i in range(10, 14):
        datas = pd.DataFrame(
            [1+i, 2+i, 3+i, 4+i, 5+i, 6+i, 7+i, 8+i, 9+i], columns=['第{}次'.format(i)])
        write2excel(excel_path, datas, sheet)
