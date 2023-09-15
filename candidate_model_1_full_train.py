# Import scipy version 1.10.1 for the stats.mode function
# to work without throwing an error. This is due to an old
# local version used during initial coding.
!pip install scipy==1.10.1

# import packages needed
import pandas as pd
import os
import numpy as np
import torch

# Set the data and output paths.
PROJECT_PATH = 'drive/MyDrive/Network Traffic Scenario/'
OUTPUT_PATH = './outputs/'
DATA_PATH = './data/'

### If using kaggle use the following setup
#PROJECT_PATH = '/kaggle/input/network-traffic-scenario/'
#DATA_PATH = PROJECT_PATH + 'data/'
#OUTPUT_PATH = '/kaggle/working/'

os.chdir(PROJECT_PATH)

# import functions and model from scripts
from help_funcs import f_feature_creation, f_train_one_epoch
from models import unet_model

# Define model name
model_id = 'candidate_model_1_full_train'

### Set the seeds for reproducibility 

rand_seed = 100
input_length = 2500

torch.manual_seed(rand_seed**3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
np.random.seed(rand_seed)


### Get full length of all training sequences

dict_seq_l = {}
for csv_id in range(78):
    
    df_seq = pd.read_csv(DATA_PATH + 'Train_data/Train' + str(csv_id) +'.csv')
    dict_seq_l[csv_id] = len(df_seq)

    
### Use all series for training; some arbitrary small 
### "validation set" for the code to run, as it does not run with an
### empty validation set

train_seqs = np.arange(78)
valid_seqs = [0,1,2]

### Training parameters and model initialization

model_device = torch.device('cpu')

model_params = {'input_features': 4,
                'no_blocks': 3,
                'cat_features': True, 
                'use_skip': True,
                'enc_channels': [16,16,32], 
                'enc_activation': torch.nn.ReLU(),
                'enc_kernel_size': [11,11,11], 
                'enc_no_layers': 2, 
                'enc_pooling': 'max_pool', 
                'enc_pooling_stride': [5,5,5],
                'enc_use_norm': True,
                'trans_layers': 3,
                'trans_channels': [32,32,32],
                'trans_kernel_size':[11,11,11],
                'trans_activation': torch.nn.ReLU(),
                'dec_channels': [16,16,16], 
                'dec_up_output_channels': [16,16,16], 
                'dec_conv_input_channels': [64,32,32],
                'dec_activation_func': torch.nn.ReLU(), 
                'dec_kernel_size': [11,11,11], 
                'dec_no_layers': 1, 
                'dec_upconv_type': 'conv_transpose', 
                'dec_upconv_stride': [5,5,5], 
                'dec_use_norm': True,
                'output_channels': 12, 
                'output_func': torch.nn.Identity(),
                'use_separate_input': True,
                'si_input_channels': 6,
                'si_output_channels': 2}

model = unet_model(model_params).to(model_device)

# The original unet_model uses an output layer kernel_size of 3 and has no
# option to change this setting. During the competition instead of expanding
# the model class to allow different kernel sizes for the output layer 
# the following workaround was used. Of course, in a future version of 
# this model the option to determine the kernel_size on model initialization
# should be added 
model.out_layer = torch.nn.Conv1d(22,12, kernel_size = 11, padding = 5, padding_mode = 'zeros').to(model_device)


### Training parameters

batch_size = 80
learning_rate = 1e-4
no_epochs = 180
optim = torch.optim.AdamW(model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.ConstantLR(optim, factor = 1, total_iters = 0)

### Load the training time series into local memory and create the additional features
### that will be fed into the model

l_local_df = []
for csv_id in range(78):
    new_df = pd.read_csv(DATA_PATH + 'Train_data/Train' + str(csv_id) +'.csv')
    new_df = f_feature_creation(new_df, 'portPktIn_mode_ratio')
    new_df = f_feature_creation(new_df, 'portPktIn_std')
    new_df = f_feature_creation(new_df, 'qSize_std')
    new_df = f_feature_creation(new_df, 'portPktIn_time_since_zero')
    new_df = f_feature_creation(new_df, 'qSize_time_since_zero')
    new_df = f_feature_creation(new_df, 'portPktIn_peak_diff')
    new_df = f_feature_creation(new_df, 'portPktIn_mean_crossing')
    new_df['portPktIn'] =  new_df['portPktIn'] / 1e6
    new_df['portPktOut'] =  new_df['portPktOut'] / 1e6
    new_df['qSize'] = np.log(new_df['qSize'] + 1) / 10 
    l_local_df += [new_df]


# Random state for training
rand_state = np.random.RandomState(rand_seed)

# Variables to keep track of best model accuracy, the epoch of
# best performance and training/validation loss and accuracy
# across all epochs
best_acc = 0
best_epoch = 0

all_train_loss = np.zeros(no_epochs) 
all_train_acc = np.zeros(no_epochs) 
all_valid_loss = np.zeros(no_epochs) 
all_valid_acc = np.zeros(no_epochs) 

### Start training

for epoch in range(no_epochs):
    # Train for one epoch
    all_train_loss[epoch], all_train_acc[epoch], all_valid_loss[epoch], all_valid_acc[epoch] = f_train_one_epoch (model, scheduler, rand_state, batch_size, 
                                train_seqs, valid_seqs, dict_seq_l, 
                                l_local_df, input_length, model_device,
                                input_cols = ['portPktIn', 'portPktOut', 'qSize','portPktIn_mode_ratio'],
                                separate_input = ['portPktIn_std', 'qSize_std','portPktIn_time_since_zero',
                                                 'qSize_time_since_zero', 'portPktIn_peak_diff', 
                                                 'portPktIn_mean_crossing'])
    # If new best model on training data found: Save model
    if all_train_acc[epoch] > best_acc:
        best_acc = all_train_acc[epoch]
        best_epoch = epoch
        torch.save({
            'model_state_dict': model.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'valid_acc': best_acc,
            'epochs_trained': best_epoch,
            'model_id': model_id}, OUTPUT_PATH + model_id + '.pth')
    
    print(epoch)
    
# Save the training log including training and validation loss and accuracy
# Note that as for this model we use the whole training set, the validation
# metrics are not of intereset. Saving the validation metrics and using
# a validation set in the first place are features added to be able to 
# evaluate models that are not trained on the full data set. 
df_save = pd.DataFrame({ 
                    'epoch': np.arange(1, no_epochs + 1),
                    'train_loss': all_train_loss,
                    'train_acc': all_train_acc,
                    'valid_loss': all_valid_loss,
                    'valid_acc': all_valid_acc})

df_save.to_csv(OUTPUT_PATH + 'train_log_' + model_id + '.csv', index = False)