# CSE:6250 Paper Reproduction

This a reproduction repository of the paper called " Temporal Pointwise Convolutional Networks for Length of Stay
Prediction in the Intensive Care Unit"

In this repository you will find the following file structure:
- config
- dataloader
- misc
- model
- trainer


## Config
In this folder you will find the best hyperparameters for each model and each dataset as described in the repository : github blah blah

The config files are json format and can be easily imported into any python script.

## Dataloder 

In this folder is where we keep the data reader / generator. These generators were directly taken from the Author's github with minor tweaks done to them. 

These data generators were purposely made to read large files (GBs) so we thought it would be ideal to use them.

## Misc

The miscelaneous folder contains all the other functions necessary for filtering,masking and padding. It also contained the different metrics used to test the models. 

## Model 

This folder contains the architecture of each of the models tested for reproducibility. In our case, you will find TPC, Transformer and LSTM model inside. 

## Trainer

This folder contains the trainer class wrapper that calls all the other classes described above. The trianer sets up the training configuration that involves the dataset, hyperparameters and model.





