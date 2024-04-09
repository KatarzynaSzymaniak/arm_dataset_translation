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

def run_grasp_clf(folds, data_config, temp_s, pos_folddict, grasp, s):
    acc_tbl = []

    # RUN EXP in FOLDS
    for fold in range(folds):
        d_test_grasp, pos_test = None, None
        d_train_grasp, pos_train = None, None

        for pos in data_config['pos']:
            
            # load fold trials for the position
            fold80_trials, fold20_trials = get_fold_trials(pos_folddict[pos]['trial_mappings'], 
                                                            pos_folddict[pos]['fold_dict'], fold=fold)
            
            # get data from the fold of a position
            y1, y2, d1, d2 = load_pos_data(temp_s, data_config, pos)
            d_train, y_train = get_fold_dataa(fold80_trials[:,0], y1, y2, d1, d2)
            d_test, y_test = get_fold_dataa(fold20_trials[:,0], y1, y2, d1, d2)

            # mask the data for a single grasp and aggregate across all positions
            d_train_grasp, pos_train= mask_grasp_data(data_config, y_train, d_train, grasp, pos, d_train_grasp, pos_train)
            d_test_grasp, pos_test = mask_grasp_data(data_config, y_test, d_test, grasp, pos, d_test_grasp, pos_test)


        # run the position clf for every grasp 
        acc = position_clf(d_train_grasp.reshape(d_train_grasp.shape[0], -1),
                            d_test_grasp.reshape(d_test_grasp.shape[0], -1),
                            pos_train,
                            pos_test)
        acc_tbl.append(acc)
        print('Subject:', s, 'Grasp:', grasp, 'Fold:', fold, 'Acc:', acc)
    return acc_tbl


def run_exp(data_config, exp_name, sub_dir, exp_config, exp_dir, folds = 5):
    results = []
    for s in exp_config['subs']:

        temp_s = os.path.join(sub_dir, 'participant_'+s)
        pos_folddict = make_pos_folds_dict(data_config, temp_s)
        print('vv', len(data_config['class'])+1)
        for grasp in range(1, len(data_config['class'])+1): # grasps are from 1 to 6
            acc_tbl = run_grasp_clf(folds, data_config, temp_s, pos_folddict, grasp, s)

            mean = np.mean(np.array(acc_tbl))
            results.append({'subject':s, 'grasp': grasp, 'acc': acc_tbl, 'mean': mean})

    df = pd.DataFrame(results)    
    df.to_csv(exp_name, index=False)

    mean_std = df.groupby(['grasp'])['mean'].agg(['mean', 'std'])
    mean_std.to_csv('subs_meanstd_'+ exp_name, index=True)


''''

def run_exp(data_config, exp_name, sub_dir, exp_config, exp_dir, folds = 5):
    

    # sub_name = os.path.basename(s)
    results = {}
    for s in exp_config['subs']:
        temp_s = os.path.join(sub_dir, 'participant_'+s)
        # load data from all positions

        pos_folddict = make_pos_folds_dict(data_config, temp_s)

        for grasp in range(1, 7):
            # RUN EXP in FOLDS
            acc_tbl = []
            for fold in range(folds):
                d_test_grasp, pos_test = None, None
                d_train_grasp, pos_train = None, None

                for pos in data_config['pos']:
                    y1, y2, d1, d2 = load_pos_data(temp_s, data_config, pos)
                    # load fold trials for the position
                    fold80_trials, fold20_trials = get_fold_trials(pos_folddict[pos]['trial_mappings'], 
                                                                   pos_folddict[pos]['fold_dict'], fold=fold)
                    
                    # get data from the fold of a position
                    d_train, y_train = get_fold_dataa(fold80_trials[:,0], y1, y2, d1, d2)
                    d_test, y_test = get_fold_dataa(fold20_trials[:,0], y1, y2, d1, d2)

                    # mask the data for a single grasp and aggregate across all positions
                    d_train_grasp, pos_train= mask_grasp_data(data_config, y_train, d_train, grasp, pos, d_train_grasp, pos_train)
                    d_test_grasp, pos_test = mask_grasp_data(data_config, y_test, d_test, grasp, pos, d_test_grasp, pos_test)


                # run the position clf for every grasp 
                acc = position_clf(d_train_grasp.reshape(d_train_grasp.shape[0], -1),
                                   d_test_grasp.reshape(d_test_grasp.shape[0], -1),
                                   pos_train,
                                   pos_test)
                acc_tbl.append(acc)
                print('Subject:', s, 'Grasp:', grasp, 'Fold:', fold, 'Acc:', acc)
            mean = np.mean(np.array(acc_tbl))
            # run the position clf for every grasp 
            results.update({'subject':s, 'grasp': grasp, 'acc': acc_tbl, 'mean': mean})

    df = pd.DataFrame(results)    
    df.to_csv(exp_name, index=False)

    mean_std = df.groupby(['grasp'])['mean'].agg(['mean', 'std'])
    mean_std.to_csv('subs_meanstd_'+exp_name, index=True)
    '''

def mask_grasp_data(data_config, y_train, d_train, grasp, pos, d_train_grasp, pos_trained):
    mask = np.where(y_train[:, 2] == grasp)[0]
    # stack the data together
    if pos == data_config['pos'][0]:
        d_train_grasp = d_train[mask]
        # y_train_grasp = y_train[mask]
        pos_trained = np.full((d_train[mask].shape[0],1), pos)
        print('unique for pos', pos, torch.unique(y_train[mask, 2]))
    else:
        d_train_grasp = np.vstack((d_train_grasp, d_train[mask]))
        # y_train_grasp = np.vstack((y_train_grasp, y_train[mask]))
        print('unique for pos', pos, torch.unique(y_train[mask, 2 ]))
        pos_train = np.full((d_train[mask].shape[0],1), pos)
        pos_trained = np.vstack((pos_trained, pos_train))
    return d_train_grasp, pos_trained


def make_pos_folds_dict(data_config, temp_s):
    pos_data = {}
    for pos in data_config['pos']:
        y1, y2, d1, d2 = load_pos_data(temp_s, data_config, pos)
        fold_dict, trial_mappings = get_fold_dict(y1, y2)
        pos_data[pos] = {'fold_dict': fold_dict, 'trial_mappings': trial_mappings}
    return pos_data



def main():

    root = os.getcwd()
    sub_dir = os.path.join(root, 'at_source', 'processed_data')

    #  -----------------  Exp Configs -----------------
    exp_config_path = os.path.join(root, 'at_source', 'configs', 'exp_TL_config.yaml')	
    exp_config = get_configs(exp_config_path)

    # #  -----------------  CONFIG + -----------------
    data_config_path = os.path.join(root, 'at_source', 'configs', '+.yaml')
    data_config = get_configs(data_config_path)

    exp_name = 'pos_clf_+config.csv'
    exp_dir = None
    run_exp(data_config, exp_name, sub_dir, exp_config, exp_dir, folds = 5)


    #  -----------------  CONFIG x -----------------
    data_config_path = os.path.join(root, 'at_source', 'configs', 'x.yaml')
    data_config = get_configs(data_config_path)

    exp_name = 'pos_clf_xconfig.csv'
    exp_dir = None
    run_exp(data_config, exp_name, sub_dir, exp_config, exp_dir, folds = 5)


if __name__ == '__main__':
    main()