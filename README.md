# CSE:6250 Paper Reproduction
## Authors: Giovanni Tobar & Ashutosh Joshi

This a reproduction repository of the paper called " Temporal Pointwise Convolutional Networks for Length of Stay
Prediction in the Intensive Care Unit"

In this repository you will find the following file structure:
```
├── config
|  └── model_name_config_eicu.py
|  └── model_name_config_mimic.py
├── dataloder
|  └── eicu_reader.py
|  └── mimic_reader.py
├── misc
|  └── utils.py
├── model
|  └── lstm.py
|  └── transformer.py
|  └── tpc.py
├── trainer
|  └── pl_trainer.py
├── dataset

```

## Config
This folder contains both the hyperparameters and configurations necessary for training each model on either eICU or MIMIC data.
The config files are in json format for human readability and can be easily imported as a dictionary.

## Dataloder 

This folder contains the data generator. once the data is processed in a train/val/test format the data loader will generate the data from these folders and feed it into the model.
## Misc

This folder contains all other functions that are outside of the main folder structure. In in here you will find functions that do filtering, masking and padding. As well as the metrics used for Mortality , LoS and Multitask

## Model 

This folder contains the architecture of the TPC, Transformer and LSTM models. 

## Trainer

This folder contains the trainer class that wraps all the other classes into one in order to read data, load data, train/val/test.
## Dataset
Dataset is a publicly available in the physionet.org, you just need to obtain a certification for eICU and MIMIC dataset.

# How to Run
To run any of the models on the either dataset you will need to open **ModelsTestBench** notebook. This notebook contains all the steps in order for you to start your own run and generate performance metrics on the desired model. 

### Requirements:
```
easydict==1.9
jupyter==1.0.0
scikit-learn== 0.24.2
scipy==1.7.1
seaborn==0.11.2
pytorch-lightning==1.6.0
torch==1.11.0
numpy==1.20.3
pandas==1.3.4
```
### References
Original Authors Repo : https://github.com/EmmaRocheteau/TPC-LoS-prediction 

