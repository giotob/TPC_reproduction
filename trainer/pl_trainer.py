import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from misc.utils import remove_padding
from model.transformer import TransformerModel
from model.tpc import TPC
from model.lstm import LSTMmodel
from model.losses import MSELoss
from torch.nn import BCELoss





class Trainer(pl.LightningModule):
    def __init__(self,cfg):
        super().__init__()
        self.lr = cfg.lr
        self.model = self.choose_model(cfg)
            
        self.test_y_list =[]
        self.test_pred_list = []
        self.test_y_mort_list =[]
        self.test_pred_mort_list = []
        

        self.cfg = cfg

        self.msle = MSELoss(log_flag=True)
        self.mse = MSELoss()
        self.bce = BCELoss()
   
    def forward(self,x,flat):
        return self.model(x,flat)
    
    def shared_step(self,batch,batch_idx):
        padded, mask, flat, los_labels, mort_labels, seq_lengths = batch
        out_los,out_mort = self(padded,flat)
        return out_los,out_mort,los_labels,mort_labels,mask,seq_lengths
        
    def training_step(self,batch, batch_idx):
        out_los,out_mort,y_los,y_mort,mask,seq_lengths = self.shared_step(batch,batch_idx)
        loss = self.loss_function(out_los,out_mort,y_los,y_mort,mask.type(torch.BoolTensor),seq_lengths)

        self.log('train_loss' ,loss,on_step=True,prog_bar=True)
        return loss
    
    def validation_step(self,batch,batch_idx):

        out_los,out_mort,y_los,y_mort,mask,seq_lengths = self.shared_step(batch,batch_idx)
        loss = self.loss_function( out_los,out_mort,y_los,y_mort,mask.type(torch.BoolTensor),seq_lengths)
        self.log('val_loss' ,loss,on_step=True,prog_bar=True)

    def test_step(self,batch,batch_idx):
        out_los,out_mort,y_los,y_mort,mask,seq_lengths = self.shared_step(batch,batch_idx)
        
        if self.cfg.task in ['multi','los']:
            self.test_y_list.append(remove_padding(y_los, mask.type(torch.BoolTensor)))
            self.test_pred_list.append(remove_padding(out_los, mask.type(torch.BoolTensor)))

        if self.cfg.task in ['multi','mort'] and (y_mort.shape[1] > 24):
            self.test_y_mort_list.append(remove_padding(y_mort, mask.type(torch.BoolTensor)))
            self.test_pred_mort_list.append(remove_padding(out_mort, mask.type(torch.BoolTensor)))
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= self.lr)
        return optimizer
    
    def loss_function(self,preds_los,preds_mort,y_los,y_mort,mask,seq_lengths):
        ls_f1 = 0
        ls_f2 = 0
        if self.cfg.task == 'multi': 
            
            if self.cfg.loss_name =='msle':
                ls_f1 = self.msle(preds_los,y_los,mask.type(torch.BoolTensor),seq_lengths)
                ls_f2 = self.bce(preds_mort,y_mort) * self.cfg.alpha

            elif self.cfg.loss_name == 'mse':
                ls_f1 = self.mse(preds_los,y_los,mask.type(torch.BoolTensor),seq_lengths)
                ls_f2 = self.bce(preds_mort,y_mort) * self.cfg.alpha
        
        elif self.cfg.task == 'los':
            
            if self.cfg.loss_name =='msle':
                ls_f1 = self.msle(preds_los,y_los,mask.type(torch.BoolTensor),seq_lengths)
            elif self.cfg.loss_name == 'mse':
                ls_f1 = self.mse(preds_los,y_los,mask.type(torch.BoolTensor),seq_lengths)

        elif self.cfg.task == 'mort':
            ls_f2 = self.bce(preds_mort,y_mort)

        return ls_f1 + ls_f2

    @staticmethod
    def choose_model(cfg):
        if  cfg.model_name =='tpc':
            return TPC(cfg)
        elif cfg.model_name == 'transformer':
            return TransformerModel(cfg)
        elif cfg.model_name =='lstm':
            return LSTMmodel(cfg)


    