import numpy as np
import os
import sys
import time
import torch
from at_source.data_utils.cv_split import *
from at_source.modelling.lda import *
from at_source.modelling.hc import *
from at_source.data_utils.utils import *
from at_source.configs.utils import get_configs
import pandas as pd
# pos = '2'
# positions = None


def position_clf(x_train, x_test, z_train, z_test):
    '''
    Using data only from a single grasp across all the positions, caluclate the accuracy of the position classifier.

    Create separate training routine for that. Still inlcude the k-fold cross validation.

    The model is using lda clf to predict the position of the object.
    '''
    
    z_train = map_pos(z_train)
    z_test = map_pos(z_test)

    acc = lda_acc(x_test, z_test, x_train, z_train)
    return acc


def run_exp(data_config, exp_name, sub_dir, exp_config, exp_dir, folds = 5):
    

    # sub_name = os.path.basename(s)
    results = {}
    for s in exp_config['subs']:
        temp_s = os.path.join(sub_dir, 'participant_'+s)
        # load data from all positions

        fold_pos_dict = make_pos_folds_dict(data_config, temp_s)

        for grasp in range(1, 7):
            # RUN EXP in FOLDS
            for fold in range(folds):
                for pos in data_config['pos']:
                    y1, y2, d1, d2 = load_pos_data(temp_s, data_config, pos)
                    # load fold settings
                    fold80_trials, fold20_trials = get_fold_trials(pos_data[pos]['trial_mappings'], 
                                                                   pos_data[pos]['fold_dict'], fold=fold)
                    d_train, y_train = get_fold_dataa(fold80_trials[:,0], y1, y2, d1, d2)
                    d_test, y_test = get_fold_dataa(fold20_trials[:,0], y1, y2, d1, d2)

                    # Single grasp data, aggregate across all positions
                    d_train_grasp, y_train_grasp, pos_train= mask_grasp_data(data_config, y_train, d_train, grasp)
                    d_test_grasp, y_test_grasp, pos_test = mask_grasp_data(data_config, y_test, d_test, grasp)

                
            # run the position clf for every grasp 
            acc = position_clf(d_train_grasp, d_test_grasp, pos_train, pos_test)
            mean = None
            # run the position clf for every grasp 
            results.append({'subject':s, 'grasp': grasp, 'acc': acc, 'mean': mean})

    df = pd.DataFrame(results)    
    df.to_csv(exp_name, index=False)

    mean_std = df.groupby(['source', 'target'])['mean'].agg(['mean', 'std'])
    mean_std.to_csv('subs_meanstd_'+exp_name, index=True)

def make_pos_folds_dict(data_config, temp_s):
    pos_data = {}
    for pos in data_config['pos']:
        y1, y2, d1, d2 = load_pos_data(temp_s, data_config, pos)
        fold_dict, trial_mappings = get_fold_dict(y1, y2)
        pos_data[pos] = {'fold_dict': fold_dict, 'trial_mappings': trial_mappings}
    return pos_data


def mask_grasp_data(data_config, y_train, d_train, grasp, pos, pos_train):
    mask = np.where(y_train[:, 2] == grasp)
    # stack the data together
    if pos == data_config['pos'][0]:
        d_train_grasp = d_train[mask]
        y_train_grasp = y_train[mask]
        pos_train = np.full((d_train_grasp.shape[0],1), pos)
    else:
        d_train_grasp = np.vstack((d_train_grasp, d_train[mask]))
        y_train_grasp = np.vstack((y_train_grasp, y_train[mask]))

        pos_train = np.full((d_train_grasp.shape[0],1), pos)
        pos_train = np.vstack((pos_train, pos_train))
    return d_train_grasp, y_train_grasp, pos_train

        

'''

def run_exp(data_config, exp_name, sub_dir, exp_config, exp_dir, folds = 5):
    
    # sub_name = os.path.basename(s)
    for s in exp_config['subs']:
        temp_s = os.path.join(sub_dir, 'participant_'+s)
        # load data from all positions
        for grasp in range(1, 7):
            for pos in data_config['pos']:
                
                y1, y2, d1, d2 = load_pos_data(temp_s, data_config, pos)
                fold_dict, trial_mappings = get_fold_dict(y1, y2)


                for fold in range(folds):

                    unique_temp_y = pos # make sure is an int

                    # load fold settings
                    fold80_trials, fold20_trials = get_fold_trials(trial_mappings, fold_dict, fold=fold)
                    d_train, y_train = get_fold_dataa(fold80_trials[:,0], y1, y2, d1, d2)

                    # get only the data from a single grasp
                    mask = np.where(y_train[:, 2] == grasp)
                    d_train_grasp = d_train[mask]
                    y_train_grasp = y_train[mask]




        # run the position clf

        position_clf(d_train_grasp, x_test, y_train_grasp, z_test)


        


'''

def main():

    root = os.getcwd()
    sub_dir = os.path.join(root, 'at_source', 'processed_data')

    #  -----------------  Exp Configs -----------------
    exp_config_path = os.path.join(root, 'at_source', 'configs', 'exp_TL_config.yaml')	
    exp_config = get_configs(exp_config_path)

    #  -----------------  CONFIG + -----------------
    #TODO: exp_fn=None
    data_config_path = os.path.join(root, 'at_source', 'configs', '+.yaml')
    data_config = get_configs(data_config_path)

    exp_name = None
    exp_dir = None
    run_exp(data_config, exp_name, sub_dir, exp_config, exp_dir, folds = 5)



if __name__ == '__main__':
    main()