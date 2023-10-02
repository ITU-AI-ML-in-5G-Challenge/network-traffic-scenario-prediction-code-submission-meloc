import pandas as pd
import numpy as np
import torch
from scipy import stats

'''
f_load subseqs: Loads subsequences of a given length of the training data

    INPUTS:
        
        l_seqs:             List of the subsequences to load; each element is a 3-tuple of 
                            (csv_file_id, start index of subsequence, end index of subsequence)
                            
        seq_length:         Length of the subsequences to be loaded (start and end indices 
                            must fit this length)
        
        PATH:               PATH to the folder containing the csv-files
        
        cols:               The feature columns that should be loaded
        
        load_target:        Boolean variable; if true a separate array of the corresponding
                            targets of each subsequence is returned
                            
        load_from_memory:   If true, load the sequences from l_local_df, if False, load from
                            path specified in PATH
                            
    OUTPUTS:
        
        x:                  Numpy-Array of shape (len(l_seqs),seq_length,len(cols)) containing
                            the loaded subsequences
        y:                  If load_target = TRUE, contains the corresponding targets of shape 
                            (len(l_seqs, seq_length))
'''


def f_load_subseqs (l_seqs, seq_length, PATH, cols = ['portPktIn', 'portPktOut', 'qSize'], load_target = True
                    ,load_from_memory = False, l_local_df = None):
    
    x = np.zeros((len(l_seqs), seq_length, len(cols)))
    if load_target:
        y = np.zeros((len(l_seqs), seq_length))
    else:
        y = None
        
    for i, seq in enumerate(l_seqs):
        
        csv_id, start_idx, end_idx  = seq
        
        if load_from_memory:
            df = l_local_df[csv_id]
        else:
            df = pd.read_csv(PATH + 'Train' + str(csv_id)+ '.csv')
        
        if load_target:
            y[i,:] = df.label.iloc[start_idx:end_idx]
        
        x[i,:,:] = df[cols].iloc[start_idx : end_idx,:].to_numpy()
    
    return x,y 


'''
f_feature_creation: Helper function to create a variaty of features;
                    each separate call adds a single new column of the 
                    corresponding feature to the data frame 
    
    INPUT:
        
        df:         The dataframe to which the new column will be added
        
        new_col:    String, specifying the new feature column to be created;
                    must be one of 
                    {'portPktIn_std', 'qSize_std','portPktIn_time_since_zero',
                     'qSize_time_since_zero', 'portPktIn_peak_diff', 'portPktIn_mode_ratio'
                     'portPktIn_mean_crossing'}
'''

def f_feature_creation(df, new_col):
    
    ## portPktIn_std calculates the standard deviation of column portPktIn
    ## within in the last 25 time steps
    
    if new_col == 'portPktIn_std':
        col_to_add = df['portPktIn'].rolling(25, min_periods = 1).std().fillna(0)
        df['portPktIn_std'] = col_to_add.to_numpy() / 1e6
    
    ## qSize_std calculates the standard deviation of column qSize
    ## within in the last 25 time steps
        
    if new_col == 'qSize_std':
        col_to_add = df['qSize'].rolling(25, min_periods = 1).std().fillna(0)
        df['qSize_std'] = col_to_add.to_numpy() / 1e6
    
    ## portPktIn_time_since_zero calculates the amount of time past since the last
    ## time portPktIn was zero
        
    if new_col == 'portPktIn_time_since_zero':
        x = df['portPktIn'].to_numpy()
        col_to_add = np.zeros(x.shape[0])   # initialize new column with all zeros
        zero_mask = (x == 0)                
        zero_idxs = np.where(zero_mask)[0]  # indices where the time series is zero 
        idx_to_set = np.repeat(True, x.shape[0]) # idx_to_set contains all the indices
                                                 # of values of the new column that
                                                 # are yet to be determined
        
        idx_to_set[zero_mask] = False       # indices of values where the time series
                                            # is zero don't need to be set as they 
                                            # are already zero
        
        # In case the series starts with a non-zero: Choose time_since_zero for the
        # starting indices of the series as if the value at the imaginary position -1
        # was a zero:
        
        min_idx = zero_idxs.min()
        col_to_add[:min_idx] = np.arange(min_idx)
        idx_to_set[np.arange(min_idx)] = False
        
        # Now, as the time since zero has been set for all indices that correspond
        # to zero-valued entries themselves as well as for the starting indices in
        # case the first value is not a zero, we can start the actual calculation
        # for all other indices: We will go through a while loop; each loop run
        # corresponds to a fixed distance to the last zero; all indices that 
        # are at this distance from a zero and have not been set yet, will be set
        # to have said distance as their time since zero. We keep doing this until 
        # there are no indices to be set anymore
        
        curr_dist = 1 # Starting distance from last zero
        
        while idx_to_set.sum() > 0: # while there are still indices to be set
            candidate_idxs = zero_idxs + curr_dist  # calculate the indices with current distance from a zero
            candidate_idxs = candidate_idxs[candidate_idxs < x.shape[0]] # exclude indices that are not within the sequence anymore
            candidate_idxs = candidate_idxs[idx_to_set[candidate_idxs]] # exclude indices that have been set already
            
            col_to_add[candidate_idxs] = curr_dist # all remaining indices are set to have time since zero = curr_distance
            idx_to_set[candidate_idxs] = False # exclude the indices set in this iteration from the future iterations
            
            curr_dist += 1
        
        df['portPktIn_time_since_zero'] = np.log(col_to_add + 1) / 4 # log-transform of the new column to have a more compact distribution
        
    
    ## qSize_time_since_zero calculates the amount of time past since the last
    ## time qSize was zero. The code is a copy of portPktIn_time_since_zero
    
    if new_col == 'qSize_time_since_zero':
        x = df['qSize'].to_numpy()
        col_to_add = np.zeros(x.shape[0])
        zero_mask = (x == 0)
        zero_idxs = np.where(zero_mask)[0] 
        idx_to_set = np.repeat(True, x.shape[0])
        
        idx_to_set[zero_mask] = False
        # Set start of sequence
        
        min_idx = zero_idxs.min()
        col_to_add[:min_idx] = np.arange(min_idx)
        idx_to_set[np.arange(min_idx)] = False
        
        curr_dist = 1
        
        while idx_to_set.sum() > 0:
            candidate_idxs = zero_idxs + curr_dist
            candidate_idxs = candidate_idxs[candidate_idxs < x.shape[0]]
            candidate_idxs = candidate_idxs[idx_to_set[candidate_idxs]]
            
            col_to_add[candidate_idxs] = curr_dist
            idx_to_set[candidate_idxs] = False
            
            curr_dist += 1
        
        df['qSize_time_since_zero'] = np.log(col_to_add + 1)/4
    
    ## portPktIn_mode_ratio calculates the percentage of values that have been equal
    ## to their mode; window of calculation is 25 time steps. In other words: Take
    ## the last 25 values, calculate their mode and check what ratio of these 25
    ## points is equal to the mode calculated
        
    if new_col == 'portPktIn_mode_ratio':
        x = df['portPktIn'].to_numpy()
        all_values = np.zeros((x.shape[0],25)) 
        
        ## For every point of the series we want all_values to contain the last
        ## 25 points; for the first 24 points of the series all missing values 
        ## (i.e. values before time 0) will be imputed as zero. 
        
        all_values[:,0] = x   # First value of the 25 step window is equal to the value itself.
        for i in range(1,25): # For each of the other 24 values, translate the time series by i time steps
            all_values[i:,i] = x[:-i]
            
        # Now calculate the mode for each time step and the ratio of values equalling
        # the mode. Then, save the result as new column
        
        modes = stats.mode(all_values,1)[0] 
        mode_ratios = (np.equal(all_values, np.repeat(modes, repeats = 25, axis = 1))).mean(1)
        df['portPktIn_mode_ratio'] = mode_ratios

        
    ## portPktIn_mean_crossing calculates the mean within a window of the last 25 values
    ## and then counts how often the time series has crossed the mean value within said
    ## window of 25 time steps.
        
    if new_col == 'portPktIn_mean_crossing':
        
        x = df['portPktIn'].to_numpy()
        
        # calculate mean within a rolling window of 25 time steps.
        means = df['portPktIn'].rolling(25, min_periods = 1).mean().to_numpy() 
        crossing_counter = np.zeros((x.shape[0])) # intialize counter of mean crossings
        
        for i in range(24):
            
            # Define a and b to be the values i and (i+1) time steps ago ;
            # When doing so impute values of the series before time 0 with 0
            
            a = np.pad(x, (i,0))[:-i] if i != 0 else x
            b = np.pad(x, (i+1,0))[:-(i+1)]
            
            # Check whether from -(i+1) to -i the series crossed the mean;
            # Then add the result (1 or 0) to the mean counter
            
            bool_crossing = np.logical_or(np.logical_and(a > means,b < means), np.logical_and(a < means, b > means))
            crossing_counter += np.array(bool_crossing, dtype = int)
        
        # Normalize the results by a factor of 1/12 and save in the data frame
        df['portPktIn_mean_crossing'] = crossing_counter / 12
    
    
    ## portPktIn_peak_diff calculates the difference between the max and min of the
    ## time series within a window of the past 25 time steps.
    
    if new_col == 'portPktIn_peak_diff':
        
        col_to_add = df['portPktIn'].rolling(25, min_periods = 1).max() - df['portPktIn'].rolling(25, min_periods = 1).min()
        df['portPktIn_peak_diff'] = col_to_add.values / 1e6

        
    return df


'''
f_train_one_epoch:  Function used to train any of the created models for one epoch;
                    includes the update of the learning rate scheduler and also calculates
                    Loss and Accuracy on training and validation set after completing
                    training for the epoch.
                    

    INPUTS:
        
        model:                  The model to be trained.
        
        scheduler:              The learning rate scheduler containing the optimizer for the model.
        
        rand_state:             Numpy random state used choose subsequence offset and
                                to shuffle the training set. 
        
        batch_size:             Batch size to be used during training
        
        train_seqs:             List of indices of the series to use as training samples.
        
        valid_seqs:             List of indices of the series to use as validation samples.
        
        dict_seq_l:             Dictionary containing the total length for each series.
        
        l_local_df:             List of data frames containing all the series that have been loaded
                                previously and are stored locally in this list to speed up training.
                    
        input_length:           Length of the subsequences to be input.
        
        model_device:           The device on which to do calcualtion (torch.device('cpu') or torch.device('cuda'))
        
        label_smoothing_factor: Input parameter 'label_smoothing' for pytorchs CrossEntropyLoss.
        
        input_cols:              columns to be loaded as direct model inputs. Default = ['portPktIn', 'portPktOut', 'qSize']
        
        separate_input:         columns to be loaded as separate model inputs. Separate input columns are linearly projected
                                into a lower dimensional space by the model before being added to the direct inputs 
                                (i.e. instead of using them as direct inputs, a linear combination of less variables will be used)
                                
    
    OUTPUTS:
        
        train_loss:             The loss on the training set after finishing training for one epoch.
        
        train_acc:              The accuracy on the training set after finishing training for one epoch.
        
        valid_loss:             The loss on the validation set after finishing training for one epoch.
        
        valid_acc:              The accuracy on the validation set after finishing training for one epoch.
        
        
'''


def f_train_one_epoch (model, scheduler, rand_state, batch_size, 
                            train_seqs, valid_seqs, dict_seq_l, 
                            l_local_df, input_length, model_device,
                            label_smoothing_factor = 0.1,
                            input_cols = ['portPktIn', 'portPktOut', 'qSize'],
                            separate_input = []):
    
    all_cols = input_cols + separate_input  # all columns that need to loaded
    input_feats = len(input_cols)           # number of columns that are direct input
    
    # l_subseqs will contain the subsequences that will be fed into the model 
    # Each subsequence is characterized by the id of its csv-file, its starting
    # and ending index. Also we use a randomly generated offset to vary the training
    # samples seen by the model
    l_subseqs = []
    
    for csv_id in train_seqs:
        
        seq_offset = rand_state.randint(0,input_length)
        max_idx = dict_seq_l[csv_id]
        
        for i in range(seq_offset, max_idx - input_length, input_length):
            l_subseqs += [(csv_id, i, i + input_length)]
    
    
    rand_state.shuffle(l_subseqs) # Shuffle the subsequences for further variation of training samples
    
    no_batches = int(np.ceil(len(l_subseqs) / batch_size))
    
    ### Start epoch of training
    
    for n in range(no_batches):
        
        # iterate through l_seqs to get the subsequences chosen for every batch
        start_idx = n*batch_size
        end_idx = (n+1)*batch_size 
        l_seqs = l_subseqs[start_idx : end_idx]
        
        # load the specified columns and targets of the subsequces from l_local_df 
        x,y = f_load_subseqs(l_seqs, input_length, PATH = None, cols = all_cols
                             , load_from_memory = True, l_local_df = l_local_df)
        
        # If there is no separate input simply turn x into a torch tensor
        if separate_input == []:
        
            x = torch.permute(torch.tensor(x, dtype = torch.float32).to(model_device), (0,2,1))
            si = None
        
        # If there is a separate input divide x into si and x, to be fed separately into the model
        else:
            si = torch.permute(torch.tensor(x[:,:,input_feats:], dtype = torch.float32).to(model_device), (0,2,1))
            x = torch.permute(torch.tensor(x[:,:,:input_feats], dtype = torch.float32).to(model_device), (0,2,1)) 

        y = torch.tensor(y, dtype = torch.long).to(model_device)
        
        # Calculate model input after permuting axis 1 and 2 to have the correct shape for the models CNNs
        output = torch.permute(model(x,si), (0,2,1))
        
        # Calculate Cross Entropy Loss
        loss = torch.nn.CrossEntropyLoss(label_smoothing = label_smoothing_factor)(output.reshape(-1,12), y.reshape(-1))
        
        # Update optimizer 
        scheduler.optimizer.zero_grad()
        loss.backward()
        scheduler.optimizer.step()
    
    # Update scheduler
    scheduler.step()
    
    
    ### Evaluation on training set

    model.eval()
    
    # Arrays to save the predicted probabilities, class predictions and true labels
    all_probs_train = np.zeros((no_batches - 1, batch_size, input_length, 12))
    all_preds_train = np.zeros((no_batches - 1, batch_size, input_length))
    all_targets_train = np.zeros((no_batches - 1, batch_size, input_length))
    
    for n in range(no_batches-1):
        
        # Calculate the models output the same way as done before during training
        start_idx = n*batch_size
        end_idx = (n+1)*batch_size 
        l_seqs = l_subseqs[start_idx : end_idx]
        
        x,y = f_load_subseqs(l_seqs, input_length, PATH = None, cols = all_cols
                             , load_from_memory = True, l_local_df = l_local_df)
        
        if separate_input == []:
        
            x = torch.permute(torch.tensor(x, dtype = torch.float32).to(model_device), (0,2,1))
            si = None
        
        else:
            si = torch.permute(torch.tensor(x[:,:,input_feats:], dtype = torch.float32).to(model_device), (0,2,1))
            x = torch.permute(torch.tensor(x[:,:,:input_feats], dtype = torch.float32).to(model_device), (0,2,1)) 

        y = torch.tensor(y, dtype = torch.long).to(model_device)
        
        output = torch.permute(model(x,si), (0,2,1))
        
        # Convert the models probabilites into predictions using Softmax; Then 
        # set the models class prediction to be the class with the highest probability
        # and save probabilites, predictions and targets
        np_output_probs = (torch.nn.Softmax(2)(output)).detach().clone().cpu().numpy()
        all_probs_train[n,:,:,:] = np_output_probs
        
        output_preds = np.argmax(np_output_probs, 2)
        all_preds_train[n,:,:] = output_preds
        all_targets_train[n,:,:] = y.detach().clone().cpu().numpy()
        
    
    # Calculate loss and accuracy on the training set
    predicted_probs = all_probs_train.reshape(-1,12)[np.arange(all_targets_train.reshape(-1).shape[0]),all_targets_train.reshape(-1).astype(int)]
    train_loss = (-1) * np.log(predicted_probs).mean()
    train_acc = (all_preds_train == all_targets_train).mean()
    
    
    ### Evaluation on validation set
    
    # The same way as done during training and evaluation on the training set, 
    # we will fragment the validation series into subsequences charaterized by
    # the id of its csv_file, its start and ending index. This info is stored within
    # l_subseqs for each subsequence. The offset will be zero here and we do not
    # shuffle the list because we do not need any randomness on the validation set.
    
    l_subseqs = []

    for csv_id in valid_seqs:
        
        seq_offset = 0
        max_idx = dict_seq_l[csv_id]
        
        for i in range(seq_offset, max_idx - input_length, input_length):
            l_subseqs += [(csv_id, i, i + input_length)]

    no_batches = int(np.ceil(len(l_subseqs) / batch_size))
    
    # From here on, calculation is the same as for evaluation on training set.
    
    all_probs_valid = np.zeros((no_batches - 1, batch_size, input_length, 12))
    all_preds_valid = np.zeros((no_batches - 1, batch_size, input_length))
    all_targets_valid = np.zeros((no_batches - 1, batch_size, input_length))

    for n in range(no_batches-1):
        
        start_idx = n*batch_size
        end_idx = (n+1)*batch_size 
        l_seqs = l_subseqs[start_idx : end_idx]
        
        x,y = f_load_subseqs(l_seqs, input_length, PATH = None, cols = all_cols
                             , load_from_memory = True, l_local_df = l_local_df)
        
        if separate_input == []:
        
            x = torch.permute(torch.tensor(x, dtype = torch.float32).to(model_device), (0,2,1))
            si = None
        
        else:
            si = torch.permute(torch.tensor(x[:,:,input_feats:], dtype = torch.float32).to(model_device), (0,2,1))
            x = torch.permute(torch.tensor(x[:,:,:input_feats], dtype = torch.float32).to(model_device), (0,2,1)) 

        y = torch.tensor(y, dtype = torch.long).to(model_device)
        
        output = torch.permute(model(x,si), (0,2,1))
        
        np_output_probs = (torch.nn.Softmax(2)(output)).detach().clone().cpu().numpy()
        all_probs_valid[n,:,:,:] = np_output_probs
        
        output_preds = np.argmax(np_output_probs, 2)
        all_preds_valid[n,:,:] = output_preds
        all_targets_valid[n,:,:] = y.detach().clone().cpu().numpy()
        

    
    predicted_probs = all_probs_valid.reshape(-1,12)[np.arange(all_targets_valid.reshape(-1).shape[0]),all_targets_valid.reshape(-1).astype(int)]
    valid_loss = (-1) * np.log(predicted_probs).mean()
    valid_acc = (all_preds_valid == all_targets_valid).mean()
    
    # Set model back to training and return the training and validation metrics
    model.train()
    
    return train_loss, train_acc, valid_loss, valid_acc


def f_sliding_window_predictions(model, l_seqs, dict_seq_l, l_local_df, input_length, 
                               input_cols, model_device, return_label = True):
    
    '''
    f_sliding_window_predictions: The function takes a model and a list of time series
        IDs and makes model predictions using a sliding window across the series 
        of step size input_length/4. 
        
        This way the function ends up with 4 different predictions for each point in time.
        Each prediction consists of a vector of 12 probabilites, one for each
        class. One prediction corresponds to a window where the particular point 
        is on the outer left side of the window, one prediction corresponds to 
        a window where the point is in the center, one prediction corresponds
        to a window where the point is on the outer right side and the last
        predictions corresponds to a window where the point is "inbetween"
        the center and the outer sides (whether this is to the left or right
        of the center depends on the particular point). 
        
        The function writes these 48 probabilities into a dictionary (one entry
        for each series of the input list)
        
    
    INPUTS:
        
        -model:         The model that should be used for predictions
        
        -l_seqs:        A list containing the IDs of the csv-files of the series 
                        that should be predicted on
        
        -dict_seq_l:    A dictionary containing the full length of each time series
        
        -l_local_df:    A list containing all time series as data frames
        
        -input_length:  The input length of the model 
        
        -input_cols:    The columns that are inputs for the model
        
        -model_device:  The device the model is on and on which the calculations
                        will take place
                        
        -return_label:  Boolean; If True the data frames returned by the function 
                        have an extra column containing the labels for each time step
                        
                        
    OUTPUTS:
        
        -dict_probs:    A dictionary of dataframes for each ID in l_seqs 
                        that contain the 48 probabilities of the 4 predictions
                        for each time step as well as the target in return_label = True
    
    '''
    
    dict_probs = {}
    for csv_id in l_seqs:
        
        # First: Divide data frame into two parts: main and tail
        # main: Everything up to the last index divisible by input_length
        # tail: the last 3*input_length entries
        # This is done to be able to slide the window of size input_length
        # across the data frames
        
        full_length = dict_seq_l[csv_id]
        
        mpart_end_idx = input_length * (full_length // input_length)
        tail_start_idx = full_length - 3*input_length
        
        # mpart_df contains the main part, tail_df the tail; if return_label == True
        # the label will be copied as well
        if return_label:
            mpart_df = l_local_df[csv_id].copy().iloc[:mpart_end_idx,:][input_cols + ['label']]
            tail_df = l_local_df[csv_id].copy().iloc[tail_start_idx:,:][input_cols + ['label']]
        else:
            mpart_df = l_local_df[csv_id].copy().iloc[:mpart_end_idx,:][input_cols ]
            tail_df = l_local_df[csv_id].copy().iloc[tail_start_idx:,:][input_cols ]

        no_cols = len(mpart_df.columns)
        
        # Prepare the data frames to store the 48 predictions (12 for each 
        # window a time step is part of; these can be classified as predictions
        # where the point in time is in the center, left, right and "inbetween"
        # of the window [see also function description above])
        mpart_df[['center_pred_label_' + str(i) for i in range(12) ]] = 0
        mpart_df[['outer_left_pred_label_' + str(i) for i in range(12) ]] = 0
        mpart_df[['outer_right_pred_label_' + str(i) for i in range(12) ]] = 0
        mpart_df[['inbetween_pred_label_' + str(i) for i in range(12) ]] = 0
        
        tail_df[['center_pred_label_' + str(i) for i in range(12) ]] = 0
        tail_df[['outer_left_pred_label_' + str(i) for i in range(12) ]] = 0
        tail_df[['outer_right_pred_label_' + str(i) for i in range(12) ]] = 0
        tail_df[['inbetween_pred_label_' + str(i) for i in range(12) ]] = 0
        
        # Get the column indices of the new created columns
        cols_idx_center = np.arange(no_cols, no_cols + 12)
        cols_idx_left = np.arange(no_cols + 12, no_cols + 24)
        cols_idx_right = np.arange(no_cols + 24, no_cols + 36)
        cols_idx_inbetween = np.arange(no_cols + 36, no_cols + 48)
        
        
        # Start predicting for the tail; the tail is predicted first
        # in order for the main part to be able to overwrite the tail's
        # predictions for samples that are predicted within both parts
        
        for w_slide in range(0,input_length,int(input_length / 4)):
            
            # w_slide is the offset from the left end of the series
            # if w_slide == 0 we can predict for the full sequence,
            # if w_slide != 0 we have to cut off a part at the right end
            # to keep the sequence divisible by input_length
            if w_slide == 0:
                z = tail_df[input_cols].copy().to_numpy()
                x = torch.tensor(z[:,:4], dtype = torch.float32).to(model_device)
                si = torch.tensor(z[:,4:], dtype = torch.float32).to(model_device)
            else:
                z = tail_df.loc[(tail_start_idx + w_slide):(tail_start_idx + w_slide + 2*input_length - 1),:][input_cols].copy().to_numpy()
                x = torch.tensor(z[:,:4], dtype = torch.float32).to(model_device)
                si = torch.tensor(z[:,4:], dtype = torch.float32).to(model_device)
            
            # x is the direct model input, si is the seperate input 
            x = torch.permute(x.reshape(-1,input_length,x.shape[1]), (0,2,1))
            si = torch.permute(si.reshape(-1,input_length,si.shape[1]), (0,2,1))
            
            no_seqs = x.shape[0]
            
            # Model prediction
            output = torch.nn.Softmax(2)(torch.permute(model(x, si), (0,2,1))).detach().clone().cpu().numpy()
            output = output.reshape(-1,output.shape[2])
            
            # The following part computes the indices of tail_df that given w_slide correspond
            # to the left, right, center and inbetween part of their respective window.
            outer_left_idx = np.concatenate(list(map(lambda x: x + np.arange(w_slide, w_slide + int(input_length/4)), np.arange(0,input_length * no_seqs, input_length)))) 
            center_idx = np.concatenate(list(map(lambda x: x + np.arange(w_slide + int(input_length/8) * 3, w_slide + int(input_length/8)*5), np.arange(0,input_length * no_seqs, input_length))))
            outer_right_idx = np.concatenate(list(map(lambda x: x + np.arange(w_slide + int(input_length/4) * 3, w_slide + input_length), np.arange(0,input_length * no_seqs, input_length))))
            inbetween_idx = np.concatenate(list(map(lambda x: x + np.concatenate((np.arange(w_slide + int(input_length/4), w_slide + int(input_length/8) * 3), np.arange(w_slide + int(input_length/8)*5, w_slide + int(input_length/4) * 3))), np.arange(0,input_length * no_seqs, input_length))))
            
            # Using the indices calculated above we can store the model output in tail_df 
            # while taking the indices relative position to their window into account.
            tail_df.iloc[outer_left_idx,cols_idx_left] = output[outer_left_idx - w_slide,:]
            tail_df.iloc[center_idx,cols_idx_center] = output[center_idx - w_slide,:]
            tail_df.iloc[outer_right_idx,cols_idx_right] = output[outer_right_idx - w_slide,:]
            tail_df.iloc[inbetween_idx,cols_idx_inbetween] = output[inbetween_idx - w_slide,:]
    
        
        ## Repeat for main_part_df
    
        for w_slide in range(0,input_length,int(input_length / 4)):
            
            z = mpart_df.iloc[w_slide:(mpart_end_idx - input_length + w_slide),:][input_cols].copy().to_numpy()
            x = torch.tensor(z[:,:4], dtype = torch.float32).to(model_device)
            si = torch.tensor(z[:,4:], dtype = torch.float32).to(model_device)
            
            x = torch.permute(x.reshape(-1,input_length,x.shape[1]), (0,2,1))
            si = torch.permute(si.reshape(-1,input_length,si.shape[1]), (0,2,1))
            
            no_seqs = x.shape[0]
            
            output = torch.nn.Softmax(2)(torch.permute(model(x, si), (0,2,1))).detach().clone().cpu().numpy()
            output = output.reshape(-1,output.shape[2])
                    
            
            outer_left_idx = np.concatenate(list(map(lambda x: x + np.arange(w_slide, w_slide + int(input_length/4)), np.arange(0,input_length * no_seqs, input_length)))) 
            center_idx = np.concatenate(list(map(lambda x: x + np.arange(w_slide + int(input_length/8) * 3, w_slide + int(input_length/8)*5), np.arange(0,input_length * no_seqs, input_length))))
            outer_right_idx = np.concatenate(list(map(lambda x: x + np.arange(w_slide + int(input_length/4) * 3, w_slide + input_length), np.arange(0,input_length * no_seqs, input_length))))
            inbetween_idx = np.concatenate(list(map(lambda x: x + np.concatenate((np.arange(w_slide + int(input_length/4), w_slide + int(input_length/8) * 3), np.arange(w_slide + int(input_length/8)*5, w_slide + int(input_length/4) * 3))), np.arange(0,input_length * no_seqs, input_length))))
            
            #print(outer_right_idx.shape)
            
            mpart_df.iloc[outer_left_idx,cols_idx_left] = output[outer_left_idx - w_slide,:]
            mpart_df.iloc[center_idx,cols_idx_center] = output[center_idx - w_slide,:]
            mpart_df.iloc[outer_right_idx,cols_idx_right] = output[outer_right_idx - w_slide,:]
            mpart_df.iloc[inbetween_idx,cols_idx_inbetween] = output[inbetween_idx - w_slide,:]
        
        
        # Append the part of tail_df that contains predictions for indices not present
        # in mpart_df to mpart_df to create full_df, a data frame that contains all 
        # time steps of the original series and will be the data frame we will store
        # in the output dictionary at the end.
        
        full_df = pd.concat((mpart_df, tail_df.loc[mpart_end_idx:,:]))
        
        # Clean up: Because the window is not perfectly divisible by 8 and because of 
        #           cases on the two ends of the sequence we have to make sure all
        #           columns of full_df are filled; to do so we emply the following
        #           imputing strategy: First check in mpart_df if there is a 
        #           prediction corresponding to a different position relative to 
        #           the sliding window. If so, impute with the center prediction 
        #           if possible, if not, impute with the position that is "closest"
        #           to the position that is to be imputed.
        #           If there are no predictions in mpart_df, check in tail_df
        #           The order of imputing based on the position is the same.
        
        # all current predictions in full_df
        main_left = full_df.iloc[:,cols_idx_left].to_numpy()
        main_inbetween = full_df.iloc[:,cols_idx_inbetween].to_numpy()
        main_center = full_df.iloc[:,cols_idx_center].to_numpy()
        main_right = full_df.iloc[:,cols_idx_right].to_numpy()
        
        # all predictions of tail_df; preprend zeros to these predictions
        # so they have the same size as the predictions in full_df and
        # can be imputed 
        tail_left = np.concatenate((np.zeros((tail_start_idx,12)), tail_df.iloc[:,cols_idx_left]))
        tail_inbetween = np.concatenate((np.zeros((tail_start_idx,12)), tail_df.iloc[:,cols_idx_inbetween]))
        tail_center = np.concatenate((np.zeros((tail_start_idx,12)), tail_df.iloc[:,cols_idx_center]))
        tail_right = np.concatenate((np.zeros((tail_start_idx,12)), tail_df.iloc[:,cols_idx_right]))
        
        
        # Left imputing
        
        main_left[main_left.sum(1) == 0] = main_center[main_left.sum(1) == 0]
        main_left[main_left.sum(1) == 0] = main_inbetween[main_left.sum(1) == 0]
        main_left[main_left.sum(1) == 0] = main_right[main_left.sum(1) == 0]
        main_left[main_left.sum(1) == 0] = tail_left[main_left.sum(1) == 0]
        main_left[main_left.sum(1) == 0] = tail_inbetween[main_left.sum(1) == 0]
        main_left[main_left.sum(1) == 0] = tail_center[main_left.sum(1) == 0]
        main_left[main_left.sum(1) == 0] = tail_right[main_left.sum(1) == 0]
        
        
        # Inbetween imputing
        
        main_inbetween[main_inbetween.sum(1) == 0] = main_center[main_inbetween.sum(1) == 0]
        main_inbetween[main_inbetween.sum(1) == 0] = main_right[main_inbetween.sum(1) == 0]
        main_inbetween[main_inbetween.sum(1) == 0] = main_left[main_inbetween.sum(1) == 0]
        main_inbetween[main_inbetween.sum(1) == 0] = tail_inbetween[main_inbetween.sum(1) == 0]
        main_inbetween[main_inbetween.sum(1) == 0] = tail_center[main_inbetween.sum(1) == 0]
        main_inbetween[main_inbetween.sum(1) == 0] = tail_right[main_inbetween.sum(1) == 0]
        main_inbetween[main_inbetween.sum(1) == 0] = tail_left[main_inbetween.sum(1) == 0]
        
        
        # Center Imputing
        
        main_center[main_center.sum(1) == 0] = main_inbetween[main_center.sum(1) == 0]
        main_center[main_center.sum(1) == 0] = main_right[main_center.sum(1) == 0]
        main_center[main_center.sum(1) == 0] = main_left[main_center.sum(1) == 0]
        main_center[main_center.sum(1) == 0] = tail_center[main_center.sum(1) == 0]
        main_center[main_center.sum(1) == 0] = tail_inbetween[main_center.sum(1) == 0]
        main_center[main_center.sum(1) == 0] = tail_right[main_center.sum(1) == 0]
        main_center[main_center.sum(1) == 0] = tail_left[main_center.sum(1) == 0]
        
        
        # Right imputing
        
        main_right[main_right.sum(1) == 0] = main_center[main_right.sum(1) == 0]
        main_right[main_right.sum(1) == 0] = main_inbetween[main_right.sum(1) == 0]
        main_right[main_right.sum(1) == 0] = main_left[main_right.sum(1) == 0]
        main_right[main_right.sum(1) == 0] = tail_right[main_right.sum(1) == 0]
        main_right[main_right.sum(1) == 0] = tail_center[main_right.sum(1) == 0]
        main_right[main_right.sum(1) == 0] = tail_inbetween[main_right.sum(1) == 0]
        main_right[main_right.sum(1) == 0] = tail_left[main_right.sum(1) == 0]
          
        # Finally copy the predictions into full_df and save the data frame in dict_probs
        full_df.iloc[:,cols_idx_left] = main_left
        full_df.iloc[:,cols_idx_right] = main_right
        full_df.iloc[:,cols_idx_center] = main_center
        full_df.iloc[:,cols_idx_inbetween] = main_inbetween
        
        dict_probs[csv_id] = full_df.copy()
    
    return dict_probs
