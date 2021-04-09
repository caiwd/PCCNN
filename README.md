# PCCNN

This is a repository of code and validation data for paper [A novel probability confidence CNN model and its application in mechanical fault diagnosis].

The description of folder as follow.
1. code: running BATCH_ALL.py will automatically complete the data pre-processing, model training, comparison method calculations and result statistics, and save the results of the ten-fold cross validation in the result folder.
2. data: raw validation data for 20 Datasets A~Q.
3. model: trained PCCNN models that can be used for testing.
4. result: the result of ten-fold cross-validation, including the proposed and comparison methods.

Required python and python libraries as follow.
1. python==3.6 or 3.7.
2. pytorch==1.4.0+cu101, CUDA==10.1.
3. xlrd==1.2.0
4. scikit-learn==0.23.2
5. pandas==1.1.1
6. numpy==1.19.2
7. nptdms==0.28.0
8. pywavelets==1.1.1
9. matplotlib==2.2.5
10. progressbar==2.5
11. paramiko==2.7.2
12. openpyxl==3.0.5
13. cvxopt==1.2.5

The raw data file, trained model files and log files has been uploaded to network drive as follows.
Network drive url: https://pan.baidu.com/s/1EF3pzDWTmjUKRD4tgh_drQ
Extraction code: wz8e

Please do not hesitate to contact me if you have any queries, my e-mail address is hanym@mail.buct.edu.cn.

/The code will be open sourced after the paper is accepted/
