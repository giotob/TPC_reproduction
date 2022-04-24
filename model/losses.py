import torch
import torch.nn.functional as F

class MSELoss(torch.nn.Module):
    def __init__(self,log_flag = False):
        super().__init__()
        self.log_flag = log_flag

    def forward(self, preds, y, mask, seq_length):
        
        # compute log before applying 0s since log(0) is undefined
        device ='cpu'
        if self.log_flag: 
            preds = preds.log()
            y = y.log()
        
        # From Author's repo, set the masked predictions to 0
        # https://github.com/EmmaRocheteau/TPC-LoS-prediction
        #####################################################################
        preds = preds.where(mask.to(device), torch.zeros_like(y).to(device)) 
        y = y.where(mask.to(device), torch.zeros_like(y).to(device)) # same thing
        #####################################################################

        loss = (y - preds)**2
        loss = torch.sum(loss, dim=1) / seq_length.clamp(min=1)
        return loss.mean()


