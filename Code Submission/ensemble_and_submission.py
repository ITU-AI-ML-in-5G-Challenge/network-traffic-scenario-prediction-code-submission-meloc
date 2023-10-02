# Install scipy version 1.10.1 for the stats.mode function
# to work without throwing an error. This is due to an old
# local version used during initial coding.
!pip install scipy==1.10.1

import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy import stats
import pickle

from datetime import datetime

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


# Define list of models that will make up the ensemble
ensemble_list =  ['candidate_model_1_full_train', 'candidate_model_2_full_train',
                    'candidate_model_4_full_train', 'candidate_model_6_full_train']

num_models = len(ensemble_list)
train_seqs = np.arange(78)
rand_seed = 100


# Get the predicted probabilities on train and test set; write them into
# the dictionaries dict_model_train_probs and dict_model_test_probs

dict_model_train_probs = {}
dict_model_test_probs = {}
col_preds = ['windowed_prob_fitted_' + str(i) for i in range(12)]
#col_preds = ['pred_label_fitted_' + str(i) for i in range(12)]   # original column names used during competition

for model_name in ensemble_list:
    pickle_file = open(OUTPUT_PATH + model_name + '_fitted_window_predictions.pickle', 'rb')
    #pickle_file = open(MODEL_SAVE_PATH + model_name + '_fitted_window_predictions.pickle', 'rb') # if using kaggle
    dict_model_train_probs[model_name] = pickle.load(pickle_file)
    pickle_file.close()
    
    
    pickle_file = open(OUTPUT_PATH + model_name + '_fitted_window_predictions_test.pickle', 'rb')
    #pickle_file = open(MODEL_SAVE_PATH + model_name + '_fitted_window_predictions_Test.pickle', 'rb') # if using kaggle
    dict_model_test_probs[model_name] = pickle.load(pickle_file)
    pickle_file.close()
    

# Similar as to the script individual_models_postprocessing:
# Write the 4*12 predicted probabilities for each time step 
# of every series into one big array and estimate a logistic regression
# that ensembles these predictions to create one final ensemble 
# prediction

all_preds = {}
all_targets = np.zeros((0))
for model_id in ensemble_list:
    
    # for each model: Write the predicted probabilities of the model into
    # one large array with 12 columns
    model_array = np.zeros((0,12))
    
    for csv_id in train_seqs:
    
        curr_df = dict_model_train_probs[model_id][csv_id]
        new_probs = curr_df[col_preds]
        model_array = np.concatenate((model_array, new_probs))
        
        # During the loop over the first model: Get targets as well
        # for the model estimation
        if model_id == ensemble_list[0]:
            all_targets = np.concatenate((all_targets, curr_df['label'].to_numpy()))
    
    all_preds[model_id] = model_array


# Concatenate the arrays of each model into one full_array of 48 columns
full_array = all_preds[ensemble_list[0]]

for model_id in ensemble_list[1:]:
    
    full_array = np.concatenate((full_array, all_preds[model_id]), axis = 1)
    
# Estimate logistic regression to build the final ensemble
print("Start estimation of logistic regression for ensemble")
log_reg = LogisticRegression(random_state = rand_seed, solver = 'sag')
log_reg.fit(full_array[:,:], all_targets[:])
print("Finished estimation of logistic regression for ensemble")

# Save the logistic regression
pickle_file = open(OUTPUT_PATH + 'ensemble_' + '_'.join(ensemble_list) + '_log_reg.pickle', 'wb')
pickle.dump(log_reg, pickle_file) 
pickle_file.close()


# Use the fitted model to create the predicted probabilities of the ensemble
# for each time series of the test set and write them into dict_test_fitted_predictions

dict_test_fitted_predictions = {}

for csv_id in range(19):
    
    # For every csv_id indiv_probs_array will contain the 48 individually predicted
    # probabilites
    indiv_probs_array = np.zeros((len(dict_model_test_probs[ensemble_list[0]][csv_id]),12*len(ensemble_list)))
    
    # Go through every model and add its predicted probabilites for the current csv_id
    # to the array 
    for i, model_id in enumerate(ensemble_list):
        
        curr_probs = dict_model_test_probs[model_id][csv_id][col_preds].to_numpy()
        indiv_probs_array[:,12*i:12*(i+1)] = curr_probs
        
    # use indiv_probs_array and the logistic regression model to create the 
    # ensemble's prediction (just one single class of maximum probability); 
    # store it in curr_df and add it to the dictionary of fitted predictions
    curr_ensemble_predictions = log_reg.predict(indiv_probs_array)  
    curr_df = dict_model_test_probs[ensemble_list[0]][csv_id][['time']]
    curr_df['ensemble_prediction'] = curr_ensemble_predictions
    
    dict_test_fitted_predictions[csv_id] = curr_df.copy()  


## Here comes the final step: The predicted labels are going to be smoothed.
## When looking at the models' outputs one can see that sometimes there
## is an oscillating between predicted classes. However, we know that each
## scenario has a minimum length of 300 time steps, hence fast oscillation 
## between scenarios does not occurr. To take this into account we smooth
## the ensembles predictions by replacing the prediction at each time step
## by a majority vote of all series indices that are 75 or less steps away
## from the current index (i.e. a majority vote among 151 predictions)
## The result will be written into the submission file as the final 
## prediction.

# radius = maximum distance for a index to be part of the vote
radius = 75
neighbour_w_size = 2*radius + 1 # full window size for vote = 151

# Load submission file
df_submission = pd.read_csv(DATA_PATH + 'SampleSubmission.csv')

# Go through every series and do the majority vote
for test_seq in range(19):
    
    curr_df = dict_test_fitted_predictions[test_seq]
    full_length = len(curr_df)
    
    # Get the ensembles' predictions
    fitted_pred = curr_df.ensemble_prediction.to_numpy()
    
    # all_values will contain the 151 predictions within a radius of 75 time steps
    # of each index
    all_values = np.zeros((fitted_pred.shape[0],neighbour_w_size))
    
    all_values[:,0] = fitted_pred
    for i in range(1,radius + 1):
        all_values[i:,i] = fitted_pred[:-i]
        all_values[:i,i] = fitted_pred[0]
        
        all_values[:-i,(i+radius)] = fitted_pred[i:]
        all_values[-i:,(i+radius)] = fitted_pred[-1]
    
    # Among the 151 predictions make a majority vote by calculating the mode
    modes = stats.mode(all_values,1)[0].reshape(-1)
    
    # Finally write the predicitons into the submission file
    all_ids = ['test' + str(test_seq) + '_' + str(i) for i in range(full_length)]
        
    df_submission.loc[df_submission['ID'].isin(all_ids),'Target'] = modes

df_submission.to_csv(OUTPUT_PATH + 'submission_ensemble_' + '_'.join(ensemble_list) + '.csv', index = False)
