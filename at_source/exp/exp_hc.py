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
from at_source.modelling.hc import *



def run_exp(data_config, exp_name, sub_dir, exp_config, exp_dir, folds = 5):
    
    # sub_name = os.path.basename(s)
    results = []
    for s in exp_config['subs']:
        temp_s = os.path.join(sub_dir, 'participant_'+s)

        (data_cross_config, data_diag_config) = data_config 
        pos_folddict = make_pos_folds_dict(data_diag_config, temp_s)
        poscross_folddict = make_pos_folds_dict(data_cross_config, temp_s)
        pos_folddict.update(poscross_folddict) # pos 5 will be overwritten by pos 5 in cross config
        
        # RUN EXP in FOLDS
        acc_tbl = []
        for fold in range(folds):
            d_train_all = None
            

            for pos in range(1,10): #data_config['pos']:
                
                # load fold trials for the position
                fold80_trials, fold20_trials = get_fold_trials(pos_folddict[pos]['trial_mappings'], 
                                                                pos_folddict[pos]['fold_dict'], fold=fold)
                
                # get data from the fold of a position
                if pos in data_diag_config['pos']:
                    y1, y2, d1, d2 = load_pos_data(temp_s, data_diag_config, pos)
                else:
                    y1, y2, d1, d2 = load_pos_data(temp_s, data_cross_config, pos)
                d_train, y_train = get_fold_dataa(fold80_trials[:,0], y1, y2, d1, d2)
                d_test, y_test = get_fold_dataa(fold20_trials[:,0], y1, y2, d1, d2)


                # --------------------- TODO ---------------------
                # AGGREGATE THEM ALL
                if d_train_all is None:
                    d_train_all = d_train
                    y_train_all = y_train
                    d_test_all = d_test
                    y_test_all = y_test
                else:
                    d_train_all = np.concatenate((d_train_all, d_train), axis=0)
                    y_train_all = np.concatenate((y_train_all, y_train), axis=0)
                    d_test_all = np.concatenate((d_test_all, d_test), axis=0)
                    y_test_all = np.concatenate((y_test_all, y_test), axis=0)
                    
            # run the position clf for every grasp 
            acc = h_clf_acc(d_test_all.reshape(d_test_all.shape[0], -1), 
                            y_test_all[:,2], 
                            y_test_all[:,3], 
                            d_train_all.reshape(d_train_all.shape[0], -1),
                            y_train_all[:,2], 
                            y_train_all[:,3])
                    
            print('Subject:', s, 'Fold:', fold, 'Acc:', acc)                  
            acc_tbl.append(acc)
            
        mean = np.mean(np.array(acc_tbl), axis=0) # mean acc across the folds for each position
        for i, m in enumerate(mean):
            results.append({'subject': s, 'acc': acc_tbl, 'pos': i+1, 'mean_acc': m})

    df = pd.DataFrame(results)    
    df.to_csv(exp_name, index=False)  
    mean_pos_mean_std = df.groupby(['pos'])['mean_acc'].agg(['mean', 'std'])
    mean_pos_mean_std.to_csv('mean_pos_meanstd_'+exp_name, index=True)



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

    #  -----------------  DATA CONFIG -----------------
    data_crossconfig_path = os.path.join(root, 'at_source', 'configs', '+.yaml')
    data_crossconfig = get_configs(data_crossconfig_path)


    data_config_path = os.path.join(root, 'at_source', 'configs', 'x.yaml')
    data_diagconfig = get_configs(data_config_path)
    
    #  -----------------  RUN EXP. -----------------
    exp_name = 'naive_hmc_exp.csv'
    exp_dir = None
    run_exp((data_crossconfig, data_diagconfig), exp_name, sub_dir, exp_config, exp_dir, folds = 5)




if __name__ == '__main__':
    main()