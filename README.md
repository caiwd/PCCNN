# PCCNN

This is a repository of code and validation data for paper [A novel probability confidence CNN model and its application in mechanical fault diagnosis]

The description of folder:

1. code: running BATCH_ALL.py will automatically complete the data pre-processing, model training, comparison method calculations and result statistics, and save the results of the 10-fold cross validation in the result folder.

2. data: storage of raw validation data for Case I and Case II.

3. model: trained PCCNN models that can be used for direct testing.

4. result: storage of the result of 10-fold cross-validation, including the proposed and comparison methods.

Required python and python libraries:
1. python 3.7.9.
2. pytorch==1.6.0+cu101, CUDA==10.1.
3. scikit-learn==0.23.2.
4. numpy==1.19.2+mkl.
5. scipy==1.3.1.
6. npTDMS==0.28.0.
7. cvxopt==1.2.5.
8. pandas==1.1.1.
9. openpyxl==3.0.5.

If you have any questions, do not hesitate to contact me.