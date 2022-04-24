import torch

import torch.nn.functional as Functional


class TPC(torch.nn.Module):
  
  def __init__(self,cfg):
    super().__init__() 
    self.cfg = cfg
    self.get_layers()
    self.remove_none = lambda x: tuple(xi for xi in x if xi is not None) #   from repo
    self.hardtanh = torch.nn.Hardtanh(min_val=1/48, max_val=100)
    self.bn_last = torch.nn.BatchNorm1d(cfg.last_size)
    
    if cfg.task == 'multi': 
        self.out_mort = torch.nn.Linear(cfg.last_size,out_features=1)
        self.los_predict = torch.nn.Linear(cfg.last_size,1)
    elif cfg.task == 'los':
        self.los_predict = torch.nn.Linear(cfg.last_size,1)
    elif cfg.task == 'mort':
        self.out_mort = torch.nn.Linear(cfg.last_size,out_features=1)

    self.drop = torch.nn.Dropout(p=0.05)
  
  def get_layers(self):
    
    self.tcn_list = torch.nn.ModuleList([])
    self.point_list = torch.nn.ModuleList([])
    num_features = self.cfg.num_features
    no_flat_features = self.cfg.no_flat_features
    # initialize parameters
    Z = 0 # output of linear layer 
    Y = 0 # o
    Zcumm = 0 # cummulative output
    self.padding = []
    for i in range(self.cfg.num_layers):
     
      # referenced from paper and git repo: https://github.com/Al-Dailami/TPC-LoS-prediction/blob/master/models/tpc_model.py
      if i < 1 : 
        dilation =  1
        cn_input = 2*num_features
      else: 
        dilation = i * (self.cfg.kernel_size - 1)
        cn_input  = (num_features + Zcumm) *( Y + 1)
        
      self.padding.append([(self.cfg.kernel_size - 1) * dilation, 0])
      cn_output = (num_features + Zcumm) * self.cfg.temp_kernels 

      
      tcn = torch.nn.Conv1d(cn_input,cn_output, dilation = dilation, kernel_size= self.cfg.kernel_size,groups=num_features + Zcumm)

      input_dim =  (num_features + (Zcumm - Z)) * Y+ Z + 2 * num_features + self.cfg.add_features + no_flat_features
     
      output_dim = self.cfg.point_size
      pointwise  = torch.nn.Linear(input_dim,output_dim)
      
      self.tcn_list.append(tcn)
      self.point_list.append(pointwise)
    
      Y = self.cfg.temp_kernels
      Z = output_dim
      Zcumm += Z

      los_input_size = (num_features + Zcumm) * (1 + Y)  + no_flat_features
      self.to_los = torch.nn.Linear(in_features=los_input_size, out_features=self.cfg.last_size)
    
 # modified  from repo: https://github.com/Al-Dailami/TPC-LoS-prediction/blob/master/models/tpc_model.py
  def tpc_layer(self,layer_num,x, prev_linear,prev_conv, point_skip,x_orig,repeat_flat):

        Z = prev_linear.shape[1] if prev_linear is not None else 0
      
        x_pad = Functional.pad(x,self.padding[layer_num],'constant',0)
        x_temp = self.drop(self.tcn_list[layer_num](x_pad))

        x_cat = torch.cat(self.remove_none((prev_conv,prev_linear,x_orig,repeat_flat)),dim =1)
        
        x_point = self.drop(self.point_list[layer_num](x_cat))

        point_skip = torch.cat((point_skip, prev_linear.view(self.batch, self.timesteps, Z).permute(0, 2, 1)), dim=1) if prev_linear is not None else point_skip
        temp_skip = torch.cat((point_skip.unsqueeze(2),x_temp.view(self.batch, point_skip.shape[1], self.cfg.temp_kernels, self.timesteps)), dim=2)

        X_point_rep = x_point.view(self.batch, self.timesteps, self.cfg.point_size, 1).permute(0, 2, 3, 1).repeat(1, 1, (1 + self.cfg.temp_kernels), 1)  
        
        X_combined = Functional.relu(torch.cat((temp_skip, X_point_rep), dim=1))
        
        next_x = X_combined.contiguous().view(self.batch, (point_skip.shape[1] + self.cfg.point_size) * (1 + self.cfg.temp_kernels), self.timesteps) 
        temp_output = x_temp.permute(0, 2, 1).contiguous().view(self.batch * self.timesteps, point_skip.shape[1] * self.cfg.temp_kernels)
        
        return  temp_output, x_point, next_x, point_skip
    
      
  def forward(self,x,flat):
      
      self.batch,self.features, self.timesteps = x.size()
      _,self.flat_features = flat.size()
      self.cfg.time_before_pred = 5  
      self.features = self.cfg.num_features
      
      x_sep = torch.split(x[:, 1:, :], self.features, dim=1) 
      

      x_orig = x.permute(0, 2, 1).contiguous().view(self.batch * self.timesteps, -1)
     
      repeat_flat = flat.repeat_interleave(self.timesteps,dim=0) # referenced from repo
      point_skip = x_sep[0]
      conv_out = None
      linear_out = None
      
      next_x = torch.stack(x_sep, dim=2).reshape(self.batch, 2 * self.features, self.timesteps) 
      for i in range(self.cfg.num_layers):
        if i <1:
          conv_out,linear_out,next_x, point_skip = self.tpc_layer(i,next_x,linear_out,conv_out,point_skip,x_orig,repeat_flat)  
        else:
          conv_out,linear_out,next_x, point_skip = self.tpc_layer(i,next_x,linear_out,conv_out,point_skip,x_orig,repeat_flat)

      
      combined_features = torch.cat((flat.repeat_interleave(self.timesteps - self.cfg.time_before_pred, dim=0), #  from repo https://github.com/Al-Dailami/TPC-LoS-prediction/blob/master/models/tpc_model.py
                                     next_x[:, :, self.cfg.time_before_pred:].permute(0, 2, 1).contiguous().view(self.batch * (self.timesteps - self.cfg.time_before_pred), -1)), dim=1) 
      
      out = Functional.relu(self.bn_last(self.drop(self.to_los(combined_features))))
      

      if self.cfg.task =='multi':

        los_predictions = self.hardtanh(torch.exp(self.los_predict(out).view(self.batch, self.timesteps - self.cfg.time_before_pred))) # referenced from paper and repo
        mort_predictions = torch.sigmoid(self.out_mort(out).view(self.batch, self.timesteps - self.cfg.time_before_pred))  # referenced from Authors' repo
        
      elif self.cfg.task == 'los':
        los_predictions = self.hardtanh(torch.exp(self.los_predict(out).view(self.batch, self.timesteps - self.cfg.time_before_pred)))
        mort_predictions = None 

      elif self.cfg.task == 'mort':
        los_predictions = None
        mort_predictions = torch.sigmoid(self.out_mort(out).view(self.batch, self.timesteps - self.cfg.time_before_pred))  

      return los_predictions, mort_predictions
