## FICO Analytic Challenge © Fair Isaac 2024

# import the necessary libaries
import os
import sys
import copy
import time
import pandas as pd
import numpy as np
from pickle import dump, load

# Pytorch libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Sci-kit learn libraries
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Plotting library
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def modify_df(filename, path, data):
    filePath=os.path.join(path + data, filename[0] + "_features.csv")
    df1 = pd.read_csv(filePath)

    # Do not modidy
    df1['transactionDateTime'] = pd.to_datetime(df1['transactionDateTime']).astype('datetime64[ns]')
    df1 = df1.sort_values(by=['pan','transactionDateTime'])

    return df1

def matureProf_n_months(df1, datetime_col, sort_cols, n_months=2):
    # get datetime column to datetimeformat
    df1[datetime_col] = pd.to_datetime(df1[datetime_col])

    # Sort data by correct columns
    df1.sort_values(by=sort_cols, inplace=True)


    # Function to filter each pan's transactions based on its minimum date
    def filter_by_cutoff(group):
        min_date = group[datetime_col].min()
        cutoff_date = min_date + pd.DateOffset(months=n_months)
        return group[group[datetime_col] >= cutoff_date]

    # Apply the function per pan
    df1 = df1.groupby(sort_cols[0], group_keys=False).apply(filter_by_cutoff)

    # # Find earliest date in dataset
    # min_date = df1[datetime_col].min()

    # # Calculate cutoff date by adding n_months to min_date
    # cutoff_date = min_date + pd.DateOffset(months=n_months)

    # print('Earliest date: ', min_date)
    # print('Cutoff date: ', cutoff_date)

    # # Filter out rows where the datetime is less than the cutoff time
    # # df1 = df1[df1[datetime_col] >= cutoff_date]

    # # return df1
    # df1[df1[datetime_col] >= cutoff_date]

    return df1

def matureProf_n_months_boolean(df1, datetime_col, n_months=2):
    # Find earliest date in dataset
    min_date = df1[datetime_col].min()
    # Calculate cutoff date by adding n_months to min_date
    cutoff_date = min_date + pd.DateOffset(months=n_months)
    print('Earliest date: ', min_date)
    print('Cutoff date: ', cutoff_date)

    # return a boolean column which takes 'True' for rows where the datetime is less than the cutoff time, otherwise 'False'
    return df1[datetime_col] >= cutoff_date


def printMemoryUsage(df1):
    df1_mem_usage = df1.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Total Memory used by current DataFrame: {df1_mem_usage:.2f} MB")

def downcast_df(df1):
    # Downcast floats
    float_cols = df1.select_dtypes(include=['float64']).columns
    df1[float_cols] = df1[float_cols].astype(np.float32)

    # Downcast ints
    int_cols = df1.select_dtypes(include=['int64']).columns
    for col in int_cols:
        df1[col] = pd.to_numeric(df1[col], downcast='unsigned')


# Count of fraud and non-fraud accounts and transactions
def dataset_count(df1, df_isTrain=None):
    if df_isTrain is not None:
        df1 = df1[df1['is_train'] == df_isTrain]

    # Count of Unique fraud and non-fraud accounts
    accnNF_count = df1[df1['mdlIsFraudAcct'] == 0]['pan'].nunique() # Number of unique non-fraud accounts
    accnF_count = df1[df1['mdlIsFraudAcct'] == 1]['pan'].nunique() # Number of unique fraud accounts

    # Count of fraud and non-fraud transactions
    trxNF_count = len(df1[df1['mdlIsFraudTrx'] == 0]) # Number of non-fraud transactions
    trxF_count = len(df1[df1['mdlIsFraudTrx'] == 1]) # Number of fraud transactions

    if df_isTrain == 1:
        print('\033[1mTrain Set\033[0m')
    elif df_isTrain == 0:
        print('\033[1mTest Set\033[0m')
    else:
        print('\033[1mEntire Set\033[0m')
    print('# of Accounts = {:d}'.format(accnNF_count + accnF_count))
    print('# of Non-Fraud Accounts = {:d}'.format(accnNF_count))
    print('# of Fraud Accounts = {:d}'.format(accnF_count))
    print('-'*10)
    print('# of Transactions = {:d}'.format(trxNF_count + trxF_count))
    print('# of Non-Fraud Transactions = {:d}'.format(trxNF_count))
    print('# of Fraud Transactions = {:d}'.format(trxF_count))
    print('-'*10)
    print('Account Level Fraud Rate = {:.4f}'.format(accnF_count/(accnNF_count + accnF_count)))
    print('Transaction level Fraud Rate = {:.4f}'.format(trxF_count/(trxNF_count + trxF_count))) # [# of Fraud Transactions / # of All Transactions ]
    print('# of Fraud Transactions / # of Fraud Accounts = {:.2f}'.format(trxF_count/accnF_count))
    print('# of Transactions / # of Fraud Accounts = {:.2f}'.format((trxF_count + trxNF_count)/accnF_count))
    print('# of Transactions / # of Non-Fraud Accounts = {:.2f}'.format((trxF_count + trxNF_count)/accnNF_count))


def filterNFTrxfromFAccn(df1):
    if (('mdlIsFraudTrx' in df1.columns) & ('mdlIsFraudAcct' in df1.columns)):
        return df1[~((df1['mdlIsFraudTrx'] == 0) & (df1['mdlIsFraudAcct'] == 1))]
    else: 
        return df1

class MyDataset(Dataset):
    def __init__(self, x, y):
        # convert into PyTorch tensors and remember them
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        # Returns the size of the dataset
        return len(self.y)

    def __getitem__(self, idx):
        # Returns one sample from the dataset
        return self.x[idx], self.y[idx]

class NNet(torch.nn.Module):
    # Constructor; define our layers and other variables
    def __init__(self, input_size, hidden_units, output_size, dropout):
        super(NNet, self).__init__()
        self.inputLayer = nn.Linear(in_features=input_size, out_features=hidden_units)
        self.hiddenLayer = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.outputLayer = nn.Linear(in_features=hidden_units, out_features=output_size)
        self.batchNorm1 = nn.BatchNorm1d(num_features=hidden_units)
        self.batchNorm2 = nn.BatchNorm1d(num_features=hidden_units)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    # Training function; how data flows through the layers
    def forward(self, x):
        x = self.inputLayer(x)
        x = self.dropout1(self.tanh(self.batchNorm1(x)/2))
        x = self.dropout2(self.tanh(self.batchNorm2(self.hiddenLayer(x))))
        y_pred = self.sigmoid(self.outputLayer(x))
        return y_pred

def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)

def resume(model, filename):
    model.load_state_dict(torch.load(filename))

def plotROC(y_train, y_train_score, y_test, y_test_score, File1, File2, model='All Features', target_fraud_rate=0.005):
    # roc curve for models
    NF1, F1, thresh1 = roc_curve(y_train, y_train_score, pos_label=1)
    NF2, F2, thresh2 = roc_curve(y_test, y_test_score, pos_label=1)
    roc_data = {'NF1': NF1, 'F1': F1, 'NF2': NF2, 'F2': F2}
    with open(f'roc_data_test_C.pkl', 'wb') as f:
      dump(roc_data, f)
    # roc curve for tpr = fpr
    random_probs = [0 for i in range(len(y_test))]
    p_NF, p_F, _ = roc_curve(y_test, random_probs, pos_label=1)

    # plot roc curves
    plt.plot(NF1, F1, linestyle='--',color='orange', label=File1)
    plt.plot(NF2, F2, linestyle='--',color='green', label=File2)
    plt.plot(p_NF, p_F, linestyle='--', color='blue')

    if target_fraud_rate != None:
        # Find the Fraud Capture Rate at the 0.5% Non-Fraud Capture Rate
        target_NF = target_fraud_rate
        idx1 = np.argmin(np.abs(NF1 - target_NF))
        target_F1 = F1[idx1]

        idx2 = np.argmin(np.abs(NF2 - target_NF))
        target_F2 = F2[idx2]

        print(f"Fraud capture rate at {target_NF} of Frauds in train data is : {target_F1}")
        print(f"Fraud capture rate at {target_NF} of Frauds in test data is : {target_F2}")
        # Plot vertical line at target NF
        plt.axvline(x=target_NF, ymin=0, ymax=target_F1, color='red', linestyle='--', label=f'FPR = {target_NF * 100:.1f}%')
        # Plot horizontal line at corresponding F
        plt.axhline(y=target_F1, xmin=0, xmax=target_NF,color='red', linestyle='--')

        plt.xlim(0, min(target_fraud_rate*10, 1))

    # title
    plt.title('ROC curve : '+model)
    # x label
    plt.xlabel('% Non-Frauds')
    # y label
    plt.ylabel('% Frauds')

    plt.legend(loc='best')
    plt.savefig('ROC_B',dpi=300)
    plt.show();








