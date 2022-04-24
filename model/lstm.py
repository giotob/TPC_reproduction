import math
from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMmodel(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.cfg = cfg

        units = cfg.hidden_size //2 if cfg.bidirectional else cfg.hidden_size
        self.lstm = nn.LSTM(cfg.input_size,units,cfg.num_layers,bidirectional=cfg.bidirectional, dropout = cfg.dropout) # bidirectional referenced from repo

        self.drop = nn.Dropout(cfg.dropout)

        point_input = cfg.hidden_size + cfg.flat_features
        self.proj = nn.Linear(point_input,cfg.last_size)
        self.bn = nn.BatchNorm1d(cfg.last_size)
        self.out_los = nn.Linear(cfg.last_size,out_features=1)

        if cfg.task == 'multi': 
            self.out_mort = torch.nn.Linear(cfg.last_size,out_features=1)
            self.los_predict = torch.nn.Linear(cfg.last_size,1)
        elif cfg.task == 'los':
            self.los_predict = torch.nn.Linear(cfg.last_size,1)
        elif cfg.task == 'mort':
            self.out_mort = torch.nn.Linear(cfg.last_size,out_features=1)

        self.hardtanh = nn.Hardtanh(min_val=1 / 48, max_val=100) # referenced from paper and repo

    def forward(self,x,flat):
        batch,all_features,timeseries = x.size()

        h,_ = self.lstm(x.permute(2,0,1)) # From Batch,Features,Sequence -> Sequence, Batch, Features

        h = F.relu(self.drop(h.permute(1,2,0))) # Batch,Feature,sequence length
        
        # From Repo 
        combined_features = torch.cat((flat.repeat_interleave(timeseries - self.cfg.time_before_pred, dim=0),  
                                     h[:, :, self.cfg.time_before_pred:].permute(0, 2, 1).contiguous().view(batch * (timeseries - self.cfg.time_before_pred), -1)), dim=1)
         ############################   
         
        los_proj = F.relu(self.drop(self.bn(self.proj(combined_features))))

        if self.cfg.task =='multi':
            los_preds =  self.hardtanh(torch.exp(self.out_los(los_proj).view(batch, timeseries - self.cfg.time_before_pred))) # referenced from paper and repo
            mort_predictions = torch.sigmoid(self.out_mort(los_proj).view(batch, timeseries - self.cfg.time_before_pred))  #  from Authors' repo
            
        elif self.cfg.task =='los':
            los_preds =  self.hardtanh(torch.exp(self.out_los(los_proj).view(batch, timeseries - self.cfg.time_before_pred))) 
            mort_predictions = None
        elif self.cfg.task =='mort':
            los_preds = None
            mort_predictions = torch.sigmoid(self.out_mort(los_proj).view(batch, timeseries - self.cfg.time_before_pred))  
            
        return los_preds, mort_predictions
