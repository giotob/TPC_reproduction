import numpy as np
import torch
from sklearn.metrics import mean_squared_error,mean_absolute_error,cohen_kappa_score,r2_score
from sklearn.metrics import accuracy_score,auc,precision_recall_curve,classification_report,roc_auc_score
from easydict import EasyDict
import json
import pandas as pd

# https://github.com/EmmaRocheteau/TPC-LoS-prediction 
# from Authors' repo. It removes the padding for smaller timeseries to accurately calculate metrics.
def remove_padding(y, mask):
    """
        Filters out padding from tensor of predictions or labels
        Args:
            y: tensor of los predictions or labels
            mask (bool_type): tensor showing which values are padding (0) and which are data (1)
    """
    device = 'cpu' #'cuda' if torch.cuda.is_available else 'cpu'
    y = y.where(mask.to(device), torch.tensor(float('nan')).to(device)).flatten().detach().cpu().numpy()
    y = y[~np.isnan(y)]
    return y


def mean_square_log_errror(y,preds):
    return np.square(np.log(y) - np.log(preds)).mean() # From Paper

# From Author's repository, added scaling to stop MAPE from being extremely large.
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(4/24, y_true))) * 100  


# From Author's repo. This Class is for the cohen Kappa Metric 
class CustomBins:
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)

def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0]
        b = CustomBins.bins[i][1]

        if a <= x < b:
            if one_hot:
                onehot = np.zeros((CustomBins.nbins,))
                onehot[i] = 1
                return onehot
            return i
    return None


def los_metrics(y,preds):
   
    print('*'*10,'Lenght of Stay Metrics','*'*10)
    mse = mean_squared_error(y,preds)
    msle = mean_square_log_errror(y,preds)
    mae = mean_absolute_error(y,preds)
    mape = mean_absolute_percentage_error(y,preds)
    r2 = r2_score(y,preds)
    print(f'Mean Square Error: {mse}' )
    print(f'Mean Square Log Error: {msle}' )
    print(f'Mean Absolute Error: {mae}' )
    print(f'Mean Absolute Percentage Error: {mape}' )
    print(f'Rsquare: {r2}')
    
    # This metric definitely had to be taken from Author's repository. 
    # It's a metric with additional steps not mentioned in the paper.
    y_true_bins = [get_bin_custom(x, CustomBins.nbins)for x in y]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in preds]
    cohen = cohen_kappa_score(y_true_bins,prediction_bins)
    print(f'Cohen Kappa Score: {cohen}')

def mort_metrics(y,pred_mort):
    print('*'*10,'Mortality Metrics','*'*10)
    pred_mort = np.stack((1-pred_mort,pred_mort),axis=-1)
    preds = np.argmax(pred_mort,axis=1)
    print(classification_report(y,preds,target_names=['dead','alive']))
    auroc = roc_auc_score(y,pred_mort[:,1])
    precisions, recalls, thresholds = precision_recall_curve(y, pred_mort[:, 1])
    auprc = auc(recalls, precisions)
    acc = accuracy_score(y,preds)
    print(f'Accuracy:{acc}')
    print(f'AUROC: {auroc}')
    print(f'AUPRC:{auprc}')
    

def read_json(filepath):
    with open(filepath,'r') as f:
        return EasyDict(json.load(f))

