# Install scipy version 1.10.1 for the stats.mode function
# to work without throwing an error. This is due to an old
# local version used during initial coding.
!pip install scipy==1.10.1


# Import packages
import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from scipy import stats
import pickle


# Set the data and output paths.
PROJECT_PATH = 'drive/MyDrive/Network Traffic Scenario/'
OUTPUT_PATH = './outputs/'
DATA_PATH = './data/'

### If using kaggle use the following setup
#PROJECT_PATH = '/kaggle/input/network-traffic-scenario/'
#OUTPUT_PATH = '/kaggle/working/'
#MODEL_SAVE_PATH = './kaggle_outputs/'
#DATA_PATH = './data/'

os.chdir(PROJECT_PATH)

# Import functions
from help_funcs import f_feature_creation, f_sliding_window_predictions
from models import unet_model

### Preparations

rand_seed = 100
input_length = 2500
model_device = torch.device('cpu')

# Define list of models that should have their outputs postprocessed
model_name_list =  ['candidate_model_1_full_train', 'candidate_model_2_full_train',
                    'candidate_model_4_full_train', 'candidate_model_6_full_train']

# For the models listed in model_name_list, define the model parameters in order
# to instatiate a model that can load the model state dictionary.

# Parameters for model 1
dict_model_params = {
    'candidate_model_1_full_train': {'input_features': 4,
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
                    'si_output_channels': 2}}

# Parameters of model 2 are identical to parameters of model 1 except for the
# separate input
dict_model_params['candidate_model_2_full_train'] = dict_model_params['candidate_model_1_full_train'].copy()
dict_model_params['candidate_model_2_full_train']['use_separate_input'] = False
dict_model_params['candidate_model_2_full_train']['si_input_channels'] = 0
dict_model_params['candidate_model_2_full_train']['si_output_channels'] = 0

# Parameters of model 4 and model 6 are identical to parameters of model 1
dict_model_params['candidate_model_4_full_train'] = dict_model_params['candidate_model_1_full_train'].copy()
dict_model_params['candidate_model_6_full_train'] = dict_model_params['candidate_model_1_full_train'].copy()


### Output layer kernel size
dict_change_output_layer = {'candidate_model_1_full_train': True,
                           'candidate_model_2_full_train': False,
                           'candidate_model_4_full_train': True,
                           'candidate_model_6_full_train': True}

### Get full length of all training sequences

dict_seq_l = {}
for csv_id in range(78):
    
    df_seq = pd.read_csv(DATA_PATH + 'Train_data/Train' + str(csv_id) +'.csv')
    dict_seq_l[csv_id] = len(df_seq)
    
### Do the same for all training sequences

dict_seq_l_test = {}
for csv_id in range(19):
    
    df_seq = pd.read_csv(DATA_PATH + 'Test_data/Test' + str(csv_id) +'.csv')
    dict_seq_l_test[csv_id] = len(df_seq)
    
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
    
### Do the same for the test time series to be able to predict on them
### later


l_local_df_test = []
for csv_id in range(19):
    
    new_df = pd.read_csv(DATA_PATH + 'Test_data/Test' + str(csv_id) +'.csv')
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
    l_local_df_test += [new_df]
    

# Columns needed for the model inputs
input_cols = ['portPktIn', 'portPktOut', 'qSize','portPktIn_mode_ratio',
              'portPktIn_std', 'qSize_std','portPktIn_time_since_zero',
              'qSize_time_since_zero', 'portPktIn_peak_diff', 'portPktIn_mean_crossing']

   
    
### Use all series for training
train_seqs = np.arange(78)

### Load the models and write them into dict_models

dict_models = {}

for model_name in model_name_list:
        
    model = unet_model(dict_model_params[model_name]).to(model_device)
    
    if dict_change_output_layer[model_name]:
        model.out_layer = torch.nn.Conv1d(22,12, 11, padding = 5, padding_mode = 'zeros').to(model_device)
    
    model_save_state = torch.load(OUTPUT_PATH + model_name + '.pth')
    #model_save_state = torch.load(MODEL_SAVE_PATH + model_name + '.pth') # for kaggle 
    model.load_state_dict(model_save_state['model_state_dict'])
    model.eval()
    dict_models[model_name] = model
    print("Loaded model ", model_name)
    

### Start Post Processing

for model_name in model_name_list:
    

    # Step 1: Slide a window across each full train time series with a step size
    #         of input_length/4 and make predictions for each window. This way
    #         we end up with 4 different predictions for each point in time.
    #         Each prediction consists of a vector of 12 probabilites, one for each
    #         class. Hence we have 48 probabilities in total after this sliding window
    #         prediction. We save these probabilites in dict_train_probs.
    print("Start logistic regression estimation for model ", model_name)
    print("Get windowed predictions")
    dict_train_probs = f_sliding_window_predictions(dict_models[model_name], train_seqs, dict_seq_l, l_local_df, input_length, 
                                   input_cols, model_device)
    
    # Number of columns in original data frame in order to calculate column indices
    # for the different prediction locations
    no_cols = len(l_local_df[0].columns) - 1
    
    # Calculate the column indices for left, right, center and inbetween predictions
    cols_idx_center = np.arange(no_cols, no_cols + 12)
    cols_idx_left = np.arange(no_cols + 12, no_cols + 24)
    cols_idx_right = np.arange(no_cols + 24, no_cols + 36)
    cols_idx_inbetween = np.arange(no_cols + 36, no_cols + 48)
    
    
    # Step 2: Estimate a Logistic Regression Model on the 48 predicted probabilities
    #         to aggregate them to one single prediction of a 12 dimensional 
    #         probability vector, one for each class.
    
    print("Fit logistic regression to summarize windowed predictions.")
    # full_arr will contain the 48 probabilities for all time steps of all series#
    # train targets will contain the corresponding targets
    full_arr = np.zeros((0,48))
    train_targets = np.zeros((0))
    
    # Write probabilities and targets into the vector for each time series
    for csv_id in train_seqs:
        
        # Name of the columns containing the sliding window predictions 
        prediction_cols =  ['center_pred_label_' + str(i) for i in range(12) ] + ['outer_left_pred_label_' + str(i) for i in range(12) ] + ['outer_right_pred_label_' + str(i) for i in range(12) ] + ['inbetween_pred_label_' + str(i) for i in range(12) ]
        
        curr_targets = dict_train_probs[csv_id]['label'].to_numpy()
        curr_preds = dict_train_probs[csv_id][prediction_cols].to_numpy()
        full_arr = np.concatenate((full_arr, curr_preds))
        train_targets = np.concatenate((train_targets, curr_targets))
        
    # Estimate logistic regression model 
    log_reg = LogisticRegression(random_state = rand_seed, solver = 'sag')
    log_reg.fit(full_arr, train_targets)
    
    # Save the logistic regression object 
    pickle_file = open(OUTPUT_PATH +  model_name + '_lin_reg_window_summarizer.pickle', 'wb')
    pickle.dump(log_reg, pickle_file) 
    pickle_file.close()
    
    
    # Step 3: Use the logistic regression model to summarize the 4 different window 
    #         predictions into one single prediction of 12 probabilities; this is
    #         the post processed prediction corresponding to the model and we
    #         save it for further use in the final ensemble
    
    print("Save logistic regression output for training sequences for further use.")
    dict_train_fitted_probs = {}
    for csv_id in train_seqs:
        
        # Do prediction 
        prediction_cols =  ['center_pred_label_' + str(i) for i in range(12) ] + ['outer_left_pred_label_' + str(i) for i in range(12) ] + ['outer_right_pred_label_' + str(i) for i in range(12) ] + ['inbetween_pred_label_' + str(i) for i in range(12) ]
        curr_targets = dict_train_probs[csv_id]['label'].to_numpy()
        curr_preds = dict_train_probs[csv_id][prediction_cols].to_numpy()
        curr_fitted =  log_reg.predict_proba(curr_preds)
        curr_df = pd.DataFrame({'time': np.arange(curr_preds.shape[0]),
                                'label': curr_targets})
        
        # Name of the fitted probabilities
        fitted_cols = ['windowed_prob_fitted_' + str(i) for i in range(12)]
        curr_df[fitted_cols] = curr_fitted
    
        dict_train_fitted_probs[csv_id] = curr_df.copy()


    pickle_file = open(OUTPUT_PATH + model_name + '_fitted_window_predictions.pickle', 'wb')
    pickle.dump(dict_train_fitted_probs, pickle_file) 
    pickle_file.close()
    
    
    # Step 4: Do the same for the test set: Get sliding window predictions, then use
    #         the logistic regression model to estimate the postprocessed predicted
    #         probabilities of each model.
    
    print("Calculate fitted probabilities on test set for further use.")
    
    dict_test_probs = f_sliding_window_predictions(dict_models[model_name], range(19), dict_seq_l_test, l_local_df_test, input_length, 
                                   input_cols, model_device, return_label = False)
    
    dict_test_fitted_probs = {}
    for csv_id in range(19):
        
        prediction_cols =  ['center_pred_label_' + str(i) for i in range(12) ] + ['outer_left_pred_label_' + str(i) for i in range(12) ] + ['outer_right_pred_label_' + str(i) for i in range(12) ] + ['inbetween_pred_label_' + str(i) for i in range(12) ]
        curr_preds = dict_test_probs[csv_id][prediction_cols].to_numpy()
        curr_fitted =  log_reg.predict_proba(curr_preds)
        curr_df = pd.DataFrame({'time': np.arange(curr_preds.shape[0])})
        
        fitted_cols = ['windowed_prob_fitted_' + str(i) for i in range(12)]
        curr_df[fitted_cols] = curr_fitted

        dict_test_fitted_probs[csv_id] = curr_df.copy()
        
    pickle_file = open(OUTPUT_PATH + model_name + '_fitted_window_predictions_test.pickle', 'wb')
    pickle.dump(dict_test_fitted_probs, pickle_file) 
    pickle_file.close()
