## FICO Analytic Challenge © Fair Isaac 2024

# import the necessary libaries
import os
import sys
import numpy as np
import pandas as pd
from pickle import dump, load

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import math

# Pytorch libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

# Removing limitation in viewing pandas columns and rows
pd.set_option('display.max_columns', None, 'display.max_rows', None)

def modify_df(filename):
    df1 = pd.read_csv(filename)
    df1['transactionDateTime'] = pd.to_datetime(df1['transactionDateTime'])
    df1 = df1.sort_values(by=['pan','transactionDateTime'])
    return df1

# DO NOT modify the plotting function below
def plot_roc_same(yaxis, xaxis, yaxis_log, xaxis_log, xlabel, ylabel, legend=None, title=None):
    plt.figure(figsize=(10,6))

    l1 = plt.plot(xaxis, yaxis, label='NNet', linewidth=2, markersize=6)
    l2 = plt.plot(xaxis_log, yaxis_log, label='LogReg', linewidth=2, markersize=6)
    plt.ylabel(ylabel, size=12)
    plt.xlabel(xlabel, size=12)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    if legend is not None:
        plt.legend(title="{}".format(legend.capitalize()), loc='best')
    if title is not None:
        plt.title(title, size =15, weight='bold')
    return l1, l2

# DO NOT modify the plotting function below
def plot_roc_same_NNet(yaxis, xaxis, yaxis_hd, xaxis_hd, xlabel, ylabel, testFile, holdoutName, legend=None, title=None):
    plt.figure(figsize=(10,6))

    l1 = plt.plot(xaxis, yaxis, label=testFile[0], linewidth=2, markersize=6)
    l2 = plt.plot(xaxis_hd, yaxis_hd, label=holdoutName[0], linewidth=2, markersize=6)
    plt.ylabel(ylabel, size=12)
    plt.xlabel(xlabel, size=12)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    if legend is not None:
        plt.legend(title="{}".format(legend.capitalize()), loc='best')
    if title is not None:
        plt.title(title, size =15, weight='bold')

    return l1, l2

# Scaling scores
def scoring_predictions_logreg(df1, df_isTrain=False):
    scaler = MinMaxScaler(feature_range=(1, 999))

    # Converting probabilites to logOdds to get a distribution about origin (0)
    df1['log_odds'] = df1['y_preds'].apply(lambda p: np.log(0.99999/(1-0.99999)) if p == 1 else np.log(p/(1-p)))
    df1['score'] = scaler.fit_transform(df1['log_odds'].values[:, None]).astype(int)

    print("\033[1mLogReg\033[0m")

    if df_isTrain:
        print('\033[1mTrain Set\033[0m')
    elif not df_isTrain:
        print('\033[1mTest Set\033[0m')

    print("\033[1mAUC Values\033[0m")
    print("Y pred min = {}".format(df1[['y_preds']].min()[0]))
    print("Y pred max = {}".format(df1[['y_preds']].max()[0]))
    print("LogOdds min = {}".format(df1[['log_odds']].min()[0]))
    print("LogOdds max = {}".format(df1[['log_odds']].max()[0]))
    print("Score min = {}".format(df1[['score']].min()[0]))
    print("Score max = {}".format(df1[['score']].max()[0]))

    df1.drop(columns=['log_odds'], inplace= True)

def scoring_predictions_nn(df1, df_isTrain=False):
    scaler = MinMaxScaler(feature_range=(1, 999))

    # Converting probabilites to logOdds to get a distribution about origin (0)
    df1['log_odds'] = df1['y_preds'].apply(lambda p: np.log(0.99999/(1-0.99999)) if p == 1 else np.log(p/(1-p)))

    df1['score'] = scaler.fit_transform(df1['log_odds'].values[:, None]).astype(int)

    print("\033[1mNNet\033[0m")

    if df_isTrain:
        print('\033[1mTrain Set\033[0m')
    elif not df_isTrain:
        print('\033[1mTest Set\033[0m')

    print("\033[1mLAUC Values\033[0m")
    print("Y pred min = {}".format(df1[['y_preds']].min()[0]))
    print("Y pred max = {}".format(df1[['y_preds']].max()[0]))
    print("LogOdds min = {}".format(df1[['log_odds']].min()[0]))
    print("LogOdds max = {}".format(df1[['log_odds']].max()[0]))
    print("Score min = {}".format(df1[['score']].min()[0]))
    print("Score max = {}".format(df1[['score']].max()[0]))

    df1.drop(columns=['log_odds'], inplace= True)

    # DO NOT modify the plotting function below
def plot_scoreDist_same(df1, df2):
    '''NNet vs Logreg

        df1: NNet dataframe
        df2: Logreg dataframe
    '''
    plt.figure(figsize=(10,6))

    # NNet plot
    cnt_NNet, binEdge_NNet = pd.cut(df1['score'], bins=100, retbins=True)
    binCnts_NNet = cnt_NNet.value_counts().sort_index()
    binCnts_per_NNet = cnt_NNet.value_counts(normalize=True).sort_index() * 100

    cumulative_cnt_NNet = binCnts_NNet[::-1].cumsum()[::-1]
    cumulative_cnt_per_NNet = binCnts_per_NNet[::-1].cumsum()[::-1]

    plot_df_NNet = pd.DataFrame({
        "Score": binEdge_NNet[:-1],
        "Cumulative Count": cumulative_cnt_NNet.values,
        "Percentage": cumulative_cnt_per_NNet.values
    })

    # LogReg plot
    cnt_LogReg, binEdge_LogReg = pd.cut(df2['score'], bins=100, retbins=True)
    binCnts_LogReg = cnt_LogReg.value_counts().sort_index()
    binCnts_per_LogReg = cnt_LogReg.value_counts(normalize=True).sort_index() * 100

    cumulative_cnt_LogReg = binCnts_LogReg[::-1].cumsum()[::-1]
    cumulative_cnt_per_LogReg = binCnts_per_LogReg[::-1].cumsum()[::-1]

    plot_df_LogReg = pd.DataFrame({
        "Score": binEdge_LogReg[:-1],
        "Cumulative Count": cumulative_cnt_LogReg.values,
        "Percentage": cumulative_cnt_per_LogReg.values
    })
 
    sns.lineplot(x='Score', y='Percentage', data=plot_df_NNet, marker='o', label='NNet')
    sns.lineplot(x='Score', y='Percentage', data=plot_df_LogReg, marker='o', label='LogReg')
    plt.ylabel('% Transactions (Cumulative)', size=12)

    plt.xlabel('Scores', size=12)
    plt.title('Test Set Score Distribution')
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.show()

    # DO NOT modify the plotting function below
def plot_scoreDist_same_NNet(df1, df2, test, holdout):
    plt.figure(figsize=(10,6))

    # NNet plot
    cnt_NNet, binEdge_NNet = pd.cut(df1['score'], bins=100, retbins=True)
    binCnts_NNet = cnt_NNet.value_counts().sort_index()
    binCnts_per_NNet = cnt_NNet.value_counts(normalize=True).sort_index() * 100

    cumulative_cnt_NNet = binCnts_NNet[::-1].cumsum()[::-1]
    cumulative_cnt_per_NNet = binCnts_per_NNet[::-1].cumsum()[::-1]

    plot_df_NNet = pd.DataFrame({
        "Score": binEdge_NNet[:-1],
        "Cumulative Count": cumulative_cnt_NNet.values,
        "Percentage": cumulative_cnt_per_NNet.values
    })

    # NNet2 plot
    cnt_LogReg, binEdge_LogReg = pd.cut(df2['score'], bins=100, retbins=True)
    binCnts_LogReg = cnt_LogReg.value_counts().sort_index()
    binCnts_per_LogReg = cnt_LogReg.value_counts(normalize=True).sort_index() * 100

    cumulative_cnt_LogReg = binCnts_LogReg[::-1].cumsum()[::-1]
    cumulative_cnt_per_LogReg = binCnts_per_LogReg[::-1].cumsum()[::-1]

    plot_df_LogReg = pd.DataFrame({
        "Score": binEdge_LogReg[:-1],
        "Cumulative Count": cumulative_cnt_LogReg.values,
        "Percentage": cumulative_cnt_per_LogReg.values
    })

    sns.lineplot(x='Score', y='Percentage', data=plot_df_NNet, marker='o', label=test[0])
    sns.lineplot(x='Score', y='Percentage', data=plot_df_LogReg, marker='o', label=holdout[0])
    plt.ylabel('% Transactions (Cumulative)', size=12)

    plt.xlabel('Scores', size=12)
    plt.title('NNet Score Distribution')
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.show()

    # Filtering dataset; Do Not Modify
def filterNFTrxfromFAccn(df1):
    return df1[~((df1['mdlIsFraudTrx'] == 0) & (df1['mdlIsFraudAcct'] == 1))]

def matureProf_n_months(df1, datetime_col, sort_cols, n_months=2):
    # get datetime column to datetimeformat
    df1[datetime_col] = pd.to_datetime(df1[datetime_col])

    # Sort data by correct columns
    df1.sort_values(by=sort_cols, inplace=True)

    # Find earliest date in dataset
    min_date = df1[datetime_col].min()

    # Calculate cutoff date by adding n_months to min_date
    cutoff_date = min_date + pd.DateOffset(months=n_months)

    # Filter out rows where the datetime is less than the cutoff time
    df1 = df1[df1[datetime_col] >= cutoff_date]

    return df1

# count of fraud and non-fraud accounts
def dataset_count(df1, df_isTrain=False):
    if df_isTrain is not None:
        df1 = df1[df1['is_train'] == df_isTrain]

    accnNF_count = df1[df1['mdlIsFraudAcct'] == 0]['pan'].nunique() # Number of unique non-fraud accounts
    accnF_count = df1[df1['mdlIsFraudAcct'] == 1]['pan'].nunique() # Number of unique fraud accounts

    # count of fraud and non-fraud transactions
    trxNF_count = len(df1[df1['mdlIsFraudTrx'] == 0]) # number of non-fraud transactions
    trxF_count = len(df1[df1['mdlIsFraudTrx'] == 1]) # number of fraud transactions

    if df_isTrain:
        print('\033[1mTrain Set\033[0m')
    elif not df_isTrain:
        print('\033[1mTest Set\033[0m')

    print('# of Accounts = {:d}'.format(accnNF_count + accnF_count))
    print('# of Non-Fraud Accounts = {:d}'.format(accnNF_count))
    print('# of Fraud Accounts = {:d}'.format(accnF_count))
    print('-'*10)
    print('# of Transactions = {:d}'.format(trxNF_count + trxF_count))
    print('# of Non-Fraud Transactions = {:d}'.format(trxNF_count))
    print('# of Fraud Transactions = {:d}'.format(trxF_count))
    print('-'*10)
    print('Account Level Fraud Rate = {:.4f}'.format(accnF_count/(accnNF_count + accnF_count)))
    print('Transaction level Fraud Rate = {:.4f}'.format(trxF_count/(trxNF_count + trxF_count)))
    print('# of Fraud Transactions / # of Fraud Accounts = {:.4f}'.format(trxF_count/accnF_count))

def calcMetrics(df1, threshold_list, model_name, df_isTrain='train'):
    pNF_f = [] # Percent Non-Fraud, %NF, measures False Positive (FP) at transaction level
    TDR_f = [] # Transaction Detection Rate (TDR, %F)
    TVDR_f = [] # Transaction Value Detection Rate, % of correctly identified fraud trxs weighted by amount
    ApNF_f = [] # Account Percent Non-Fraud, measures False Positive (FP) at account level
    ADR_f = [] # Account Detection Rate (ADR), % of correctly identified fraud accounts
    AFPR_f = [] # Account False-Positive Ratio
    RTVDR_f = [] # Real-Time Value Detection Rate
    RR_f = [] # Review Rate, Total # of Accounts with score >= Score Threshold divided by Total # of Accounts

    if df_isTrain == 'train':
        df1 = df1[df1['is_train'] == 1]
    elif df_isTrain == 'test':
        df1 = df1[df1['is_train'] == 0]

    # count of fraud and non-fraud accounts
    accnNF_count = df1[df1['mdlIsFraudAcct'] == 0]['pan'].nunique() # Number of unique non-fraud accounts
    accnF_count = df1[df1['mdlIsFraudAcct'] == 1]['pan'].nunique() # Number of unique fraud accounts

    # count of fraud and non-fraud transactions
    trxNF_count = len(df1[df1['mdlIsFraudTrx'] == 0]) # number of non-fraud transactions
    trxF_count = len(df1[df1['mdlIsFraudTrx'] == 1]) # number of fraud transactions

    # Average fraud transaction amount of given dataset (account and transaction level)
    accnF_sum = df1.loc[(df1['mdlIsFraudAcct'] == 1) & (df1['mdlIsFraudTrx'] == 1), 'transactionAmount'].sum()
    trxAmtF = df1.loc[df1['mdlIsFraudTrx'] == 1, 'transactionAmount'].sum()

    for threshold in threshold_list:
        # Datafram with records greater than threshold
        threshold_df = df1[df1['score'] >= threshold ].copy()

        # Sort filtered data
        threshold_df = threshold_df.sort_values(by=['pan','transactionDateTime'])#.reset_index(drop=True, inplace=True)

        # Non-Fraud count >= threshold for account and transaction
        accnNF_thrsh = threshold_df[threshold_df['mdlIsFraudAcct'] == 0]['pan'].nunique() # Number of non-fraud accounts scored >= threshold
        trxNF_thrsh = len(threshold_df[threshold_df['mdlIsFraudTrx'] == 0]) # Number of non-fraud transactions scored >= threshold

        # Fraud count >= threshold for account and transaction
        accnF_thrsh = threshold_df[threshold_df['mdlIsFraudAcct'] == 1]['pan'].nunique() # Number of fraud accounts scored >= threshold
        trxF_thrsh = len(threshold_df[threshold_df['mdlIsFraudTrx'] == 1]) # Number of fraud transactions scored >= threshold

        # Transaction sum of fraud accounts >= threshold and total Transaction sum of fraud account
        accnF_thrsh_sum = threshold_df.loc[(threshold_df['mdlIsFraudAcct'] == 1) & (threshold_df['mdlIsFraudTrx'] == 1), 'transactionAmount'].sum()

        # $ amount of fraud transactions scored >= threshold and all fraud transactions
        trxAmtF_thrsh = threshold_df.loc[threshold_df['mdlIsFraudTrx'] == 1, 'transactionAmount'].sum()

        # ApNF and AFPR
        # Identify False Positive via new binary column
        threshold_df.loc[:,'isFalsePositive'] = (threshold_df['mdlIsFraudAcct'] == 0) & (threshold_df['mdlIsFraudTrx'] == 0)
        lastFP = {}
        accnNF_thrsh_AFPRcount = 0

        # Keeping track of processed accounts
        processed_accounts = set()

        for index, row in threshold_df.iterrows():
            account = row['pan']
            date = row['transactionDateTime']

            if account in processed_accounts:
                continue

            if account in lastFP:
                if (date - lastFP[account]).days <= 30:
                    accnNF_thrsh_AFPRcount += 1
                    processed_accounts.add(account)
                    continue

            if row['isFalsePositive']:
                lastFP[account] = date

        # Final Metrics
        pNF_f.append(round((trxNF_thrsh/trxNF_count)*100, 2))
        TDR_f.append(round((trxF_thrsh/trxF_count)*100, 2))
        TVDR_f.append(round((trxAmtF_thrsh/trxAmtF)*100, 2))
        ApNF_f.append(round((accnNF_thrsh_AFPRcount/accnNF_count)*100, 2))
        ADR_f.append(round((accnF_thrsh/accnF_count)*100, 2))
        RTVDR_f.append(round((accnF_thrsh_sum/accnF_sum)*100, 2))

    return pNF_f, TDR_f, TVDR_f, ApNF_f, ADR_f, RTVDR_f

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
    
    # Define the class for single layer NN
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
    
def holdout_score_NNet(filePath, scaleFile, feature_columns, label_column, device, model_l_NNet, saveFeatures,holdoutsaveCSV):
    # load the scaler
    scaler = load(open(scaleFile, 'rb'))

    # Import holdout data
    df1 = modify_df(filePath)

    # Remove Non-Fraud Transactions from Fraud Accounts
    df1 = filterNFTrxfromFAccn(df1)

    # Profile Maturation
    df1 = matureProf_n_months(df1, 'transactionDateTime', ['pan','transactionDateTime'], n_months=2)

    # Scale dataset
    df1[feature_columns] = scaler.transform(df1[feature_columns])

    # Load holdout set to Pytorch DataLoader
    eval_dataset = MyDataset(df1[feature_columns].values, df1[label_column].values)
    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False, drop_last=False)

    with torch.no_grad():
        infer_list_l = []
        for input_data, labels in eval_loader:
            output_l = model_l_NNet.forward(input_data.to(device))
            infer_list_l.append(output_l)

        # Dataset
        df1["y_preds"] = torch.concat(infer_list_l,axis=0).detach().numpy()

        # scaling predictions to a scoreout range
        scaleMM = MinMaxScaler(feature_range=(1, 999))

        # Transforming 'y_preds' to a Score outs
        df1['log_odds'] = df1['y_preds'].apply(lambda p: np.log(0.99999/(1-0.99999)) if p == 1 else np.log(p/(1-p)))
        df1['score'] = scaleMM.fit_transform(df1['log_odds'].values[:, None]).astype(int)
        df1.drop(columns=['log_odds'], inplace= True)

    # Apply inverse transformation for the input data
    df1[feature_columns] = scaler.inverse_transform(df1[feature_columns])

    # Save Data to file
    df1[saveFeatures].to_csv(holdoutsaveCSV, index=False)

    return df1


class MyBlindDataset(Dataset):
    def __init__(self, x):
        # convert into PyTorch tensors and remember them
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        # Returns the size of the dataset
        return len(self.x)

    def __getitem__(self, idx):
        # Returns one sample from the dataset
        return self.x[idx]

def blind_holdout_score_NNet(filePath, scaleFile, feature_columns, device, model_l_NNet, saveFeatures,holdoutsaveCSV):
    tags=False

    # load the scaler
    scaler = load(open(scaleFile, 'rb'))

    # Import holdout data
    df1 = modify_df(filePath)

    # removed NA's
    df1 = df1.fillna(0)

    if "mdlIsFraudTrx" in saveFeatures:
        saveFeatures.remove("mdlIsFraudTrx")

    if "mdlIsFraudAcct" in saveFeatures:
        saveFeatures.remove("mdlIsFraudAcct")
   
    if tags:
         # Remove Non-Fraud Transactions from Fraud Accounts
        df1 = filterNFTrxfromFAccn(df1)

        # Profile Maturation
        df1 = matureProf_n_months(df1, 'transactionDateTime', ['pan','transactionDateTime'], n_months=2)

    # Scale dataset
    df1[feature_columns] = scaler.transform(df1[feature_columns])

    # Load holdout set to Pytorch DataLoader
    eval_dataset = MyBlindDataset(df1[feature_columns].values)
    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False, drop_last=False)

    with torch.no_grad():
        infer_list_l = []
        for input_data in eval_loader:
            output_l = model_l_NNet.forward(input_data.to(device))
            infer_list_l.append(output_l)

        # Dataset
        df1["y_preds"] = torch.concat(infer_list_l,axis=0).detach().numpy()

        # scaling predictions to a scoreout range
        scaleMM = MinMaxScaler(feature_range=(1, 999))

        # Transforming 'y_preds' to a Score outs
        df1['log_odds'] = df1['y_preds'].apply(lambda p: np.log(0.99999/(1-0.99999)) if p == 1 else np.log(p/(1-p)))
        df1['score'] = scaleMM.fit_transform(df1['log_odds'].values[:, None]).astype(int)
        df1.drop(columns=['log_odds'], inplace= True)

    # Apply inverse transformation for the input data
    df1[feature_columns] = scaler.inverse_transform(df1[feature_columns])

    # Save Data to file
    df1[saveFeatures].to_csv(holdoutsaveCSV, index=False)

    return df1