
# Directly from repo
# https://github.com/EmmaRocheteau/TPC-LoS-prediction/blob/master/models/mean_median_model.py
# Under MIT License
# Used as sanity check for the dataloader step

import numpy as np
import pandas as pd
import torch
from misc.utils import los_metrics, remove_padding



def mean_median_model(train_batches,test_batches):
    train_y = np.array([])
    test_y = np.array([])
    
    for batch_idx, batch in enumerate(train_batches):

        # unpack batch
        
        padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
        
        train_y = np.append(train_y, remove_padding(los_labels, mask.type(torch.BoolTensor)))

    train_y = pd.DataFrame(train_y, columns=['true'])

    mean_train = train_y.mean().values[0]
    median_train = train_y.median().values[0]

    for batch_idx, batch in enumerate(test_batches):

        # unpack batch
        
        padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
        
        test_y = np.append(test_y, remove_padding(los_labels, mask.type(torch.BoolTensor)))

    test_y = pd.DataFrame(test_y, columns=['true'])

    test_y['mean'] = mean_train
    test_y['median'] = median_train

    print('Total predictions:')
    print('Using mean value of {}...'.format(mean_train))
    los_metrics(test_y['true'], test_y['mean'])
    print('Using median value of {}...'.format(median_train))
    los_metrics(test_y['true'], test_y['median'])