import os
import torch
import numpy as np
from at_source.data_utils.cv_split import *

def load_pos_data(temp_s, data_config, pos):
    y1 = torch.load(os.path.join(temp_s, data_config['dayblock'][0]+'_position_'+str(pos)+'_y.pt'))
    y2 = torch.load(os.path.join(temp_s, data_config['dayblock'][1]+'_position_'+str(pos)+'_y.pt'))
    d1 = torch.load(os.path.join(temp_s, data_config['dayblock'][0]+'_position_'+str(pos)+'.pt'))
    d2 = torch.load(os.path.join(temp_s, data_config['dayblock'][1]+'_position_'+str(pos)+'.pt'))
    return y1, y2, d1, d2

def check_blocks_data(y1, y2):
    mapping_1 = np.unique(y1[:, [0, 2]].numpy(), axis=0, return_index=True)[0]
    mapping_2 = np.unique(y2[:, [0, 2]].numpy(), axis=0, return_index=True)[0]
    if np.array_equal(mapping_1, mapping_2) is False:
        raise ValueError('The mappings are not equal')
    
def map_pos(arr):
    unique_values = np.unique(arr)
    mapping = {value: i+1 for i, value in enumerate(unique_values)}
    arr = np.vectorize(mapping.get)(arr)

    # print(arr)
    return arr


def pos_data(data_path, lbl_path):
    d1 = torch.load(data_path)
    y1 = torch.load(lbl_path)
    return d1, y1

def get_fold_dict(y1, y2):
    # trial and grasp mappings
    trial_mappings = get_trial_grasp_keys(y1)
    # split train and test folds with reference to trial_mappings
    fold_dict = trials_split(trial_mappings)
    check_blocks_data(y1, y2)
    return fold_dict, trial_mappings

# def get_fold_dicts_posclf(y1, y2):
#     trial_mappings = get_trial_grasp_keys(y1)

def get_fold_dataa(fold_trials, y1, y2, d1, d2):
    '''
    Get data from both blocks inside a single fold
    '''
    # fold_trials = fold80_trials[:,0]
    
    indices_b1 = np.where(np.isin(y1[:, 0], fold_trials))[0]
    indices_b2 = np.where(np.isin(y2[:, 0], fold_trials))[0]

    # get x,y for both blocks and run 
    d1 = d1[indices_b1]
    d2 = d2[indices_b2]
    y1 = y1[indices_b1]
    y2 = y2[indices_b2]
    # aggreate the data
    d = torch.vstack((d1, d2))
    y = torch.vstack((y1, y2))
    return d,y