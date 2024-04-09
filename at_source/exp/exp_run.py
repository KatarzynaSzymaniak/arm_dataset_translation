import numpy as np
import os
import sys
import time
import torch
from at_source.data_utils.cv_split import *
from at_source.modelling.lda import *
from at_source.modelling.hc import *
from at_source.configs.utils import get_configs
from at_source.data_utils.utils import *
import pandas as pd

# pos = '2'
# positions = None


def naive_tl(x_test, y_test, x_train, y_train):
    acc = lda_acc(x_test, y_test, x_train, y_train)
    return acc


def h_clf(x_test, y_test, z_train, x_train, y_train, z_test):
    acc = h_clf_acc(x_test, y_test, z_train, x_train, y_train, z_test)
    
    return acc


def run_source_only(y1, y2, d1, d2, exp_fn=naive_tl, folds=5):
    
    fold_dict, trial_mappings = get_fold_dict(y1, y2)
    accs=[]
    for fold in range(folds):
        fold80_trials, fold20_trials = get_fold_trials(trial_mappings, fold_dict, fold=fold)

        d_train, y_train = get_fold_dataa(fold80_trials[:,0], y1, y2, d1, d2)
        d_test, y_test = get_fold_dataa(fold20_trials[:,0], y1, y2, d1, d2)
        # run the experiment
        acc = exp_fn(d_test.reshape(d_test.shape[0], -1), y_test[:,2],
                     d_train.reshape(d_train.shape[0], -1), y_train[:,2], 
                     ) # y has 4 dim
        accs.append(acc)

    return accs, np.mean(np.array(accs)), np.std(np.array(accs))


def run_tl(y1, y2, y3, y4, d1, d2, d3, d4, exp_fn=naive_tl, folds=5):
    s_fold_dict, s_trial_mappings = get_fold_dict(y1, y2)
    t_fold_dict, t_trial_mappings = get_fold_dict(y3, y4)

    accs=[]
    for fold in range(folds):
        s_fold80_trials, s_fold20_trials = get_fold_trials(s_trial_mappings, s_fold_dict, fold=fold)
        t_fold80_trials, t_fold20_trials = get_fold_trials(t_trial_mappings, t_fold_dict, fold=fold)

        d_train, y_train = get_fold_dataa(s_fold80_trials[:,0], y1, y2, d1, d2)
        d_test, y_test = get_fold_dataa(t_fold20_trials[:,0], y3, y4, d3, d4)
        # run the experiment
        acc = exp_fn(d_test.reshape(d_test.shape[0], -1), y_test[:,2],
                     d_train.reshape(d_train.shape[0], -1), y_train[:,2], 
                     ) # y has 4 dim
        accs.append(acc)

    return accs, np.mean(np.array(accs)), np.std(np.array(accs))


def run_exp(exp_config, data_config, sub_dir, exp_fn=naive_tl, exp_name='tl_+config.csv'):

    # sub_name = os.path.basename(s)
    results = []
    for s in exp_config['subs']:
        temp_s = os.path.join(sub_dir, 'participant_'+s)
        print(f'____________    SUBJECT: {s}         _______________')

       
        for source in data_config['pos']:
            for target in data_config['pos']:
                
                y1 = torch.load(os.path.join(temp_s, data_config['dayblock'][0]+'_position_'+str(source)+'_y.pt'))
                y2 = torch.load(os.path.join(temp_s, data_config['dayblock'][1]+'_position_'+str(source)+'_y.pt'))
                d1 = torch.load(os.path.join(temp_s, data_config['dayblock'][0]+'_position_'+str(source)+'.pt'))
                d2 = torch.load(os.path.join(temp_s, data_config['dayblock'][1]+'_position_'+str(source)+'.pt'))

                if source == target:
                    acc, mean, std = run_source_only(y1, y2, d1, d2, exp_fn=exp_fn, folds=exp_config['k_folds'])

                elif source != target:
                    y3 = torch.load(os.path.join(temp_s, data_config['dayblock'][0]+'_position_'+str(target)+'_y.pt'))
                    y4 = torch.load(os.path.join(temp_s, data_config['dayblock'][1]+'_position_'+str(target)+'_y.pt'))
                    d3 = torch.load(os.path.join(temp_s, data_config['dayblock'][0]+'_position_'+str(target)+'.pt'))
                    d4 = torch.load(os.path.join(temp_s, data_config['dayblock'][1]+'_position_'+str(target)+'.pt'))
                    acc, mean, std = run_tl(y1, y2, y3, y4, d1, d2, d3, d4, exp_fn=exp_fn, folds=exp_config['k_folds'])

                results.append({'subject':s, 'source': source, 'target': target, 'acc': acc, 'mean': mean, 'std': std})

    df = pd.DataFrame(results)    
    df.to_csv(exp_name, index=False)

    mean_std = df.groupby(['source', 'target'])['mean'].agg(['mean', 'std'])
    mean_std.to_csv('subs_meanstd_'+ exp_name, index=True)
    print(mean_std)
            
                    


def main():
    #  -----------------  Experiment Functions -----------------
    # Choose what experiment do you want to run
    exp_fn1 = 'naive_tl'
    exp_fn2 = 'h_clf'

    root = os.getcwd()
    sub_dir = os.path.join(root, 'at_source', 'processed_data')

    #  -----------------  Exp Configs -----------------
    exp_config_path = os.path.join(root, 'at_source', 'configs', 'exp_TL_config.yaml')	
    exp_config = get_configs(exp_config_path)

    #  -----------------  CONFIG + -----------------
    data_config_path = os.path.join(root, 'at_source', 'configs', '+.yaml')
    data_config = get_configs(data_config_path)
    run_exp(exp_config, data_config, sub_dir, exp_fn=exp_fn2, exp_name='hc_+config.csv')

    #  -----------------  CONFIG x -----------------
    x_data_config_path = os.path.join(root, 'at_source', 'configs', 'x.yaml')
    x_data_config = get_configs(x_data_config_path)
    run_exp(exp_config, x_data_config, sub_dir, exp_fn=exp_fn2, exp_name='hc_xconfig.csv')




if __name__ == '__main__':
    main()