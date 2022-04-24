import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Postional Encoding from Authors' repo
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=14*24):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0, 2, 1)  # changed from max_len * d_model to 1 * d_model * max_len
        self.register_buffer('pe', pe)

    def forward(self, X):
        # X is B * d_model * T
        # self.pe[:, :, :X.size(2)] is 1 * d_model * T but is broadcast to B when added
        X = X + self.pe[:, :, :X.size(2)]  # B * d_model * T
        return X  # B * d_model * T


class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()

       
        self.input_embedding = nn.Linear(cfg.input_size,cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model)
        self.enc_layer = nn.TransformerEncoderLayer(cfg.d_model,cfg.num_heads,cfg.dim_feed,cfg.dropout,activation='relu')
        self.enc = nn.TransformerEncoder(self.enc_layer,cfg.num_layers)
    
    def forward(self,x):

        #x = self.input_embedding(x)
        # B,F,T -> B,T,F,-> B,F,dmodel -> B,dmode,T
        x = self.input_embedding(x.transpose(1,2)).transpose(1,2)
        x = self.pos_enc(x)
        x = self.enc(x.permute(2,0,1)) # B,dmodel,T -> T,B,dmodel
        return x.permute(1, 2, 0) # B,dmodel,T


class TransformerModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.cfg = cfg

        self.transformer = TransformerBlock(cfg)
        self.drop = nn.Dropout(cfg.dropout)

        point_input = cfg.d_model + cfg.flat_features
        self.point_los = nn.Linear(point_input,cfg.last_size)
        self.bn_last = nn.BatchNorm1d(cfg.last_size)
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

        h = self.transformer(x)
        h = F.relu(self.drop(h))
        
        # from repo
        combined_features = torch.cat((flat.repeat_interleave(timeseries - self.cfg.time_before_pred, dim=0),  
                                     h[:, :, self.cfg.time_before_pred:].permute(0, 2, 1).contiguous().view(batch * (timeseries - self.cfg.time_before_pred), -1)), dim=1)
        ################

        los_proj = F.relu(self.drop(self.bn_last(self.point_los(combined_features))))

        if self.cfg.task =='multi':
            los_preds =  self.hardtanh(torch.exp(self.out_los(los_proj).view(batch, timeseries - self.cfg.time_before_pred))) # from paper
            mort_predictions = torch.sigmoid(self.out_mort(los_proj).view(batch, timeseries - self.cfg.time_before_pred))  # referenced from  repo
            
        elif self.cfg.task =='los':
            los_preds =  self.hardtanh(torch.exp(self.out_los(los_proj).view(batch, timeseries - self.cfg.time_before_pred))) 
            mort_predictions = None
        elif self.cfg.task =='mort':
            los_preds = None
            mort_predictions = torch.sigmoid(self.out_mort(los_proj).view(batch, timeseries - self.cfg.time_before_pred))  # referenced from  repo
            
        return los_preds, mort_predictions
