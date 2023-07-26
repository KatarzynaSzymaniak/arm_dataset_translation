from operator import itemgetter
import numpy as np
from source.data.util import flatten
from source.data.rw_processed import get_grasp_list

def data_split_old(dir_list, grasp_config=['2', '4', '5', '6', '8'], split_ratio=0.7):
    # if experiment is run across domains (this function has to be iterated for each position
    train_split, test_split = [], []

    for grasp_lbl in grasp_config:

        temp_list = get_grasp_list(dir_list, grasp_lbl)
        split_size = int(np.floor(split_ratio * len(temp_list)))
        train_samples = np.random.choice(temp_list, size=split_size, replace=False)
        test_samples = [elem for elem in temp_list if elem not in train_samples]
        train_split.append(train_samples)
        test_split.append(test_samples)

    return flatten(train_split),  flatten(test_split)


def get_data_split(dir_list, label_by=['1', '2', '3', '4', '5', '6'], split_ratio=0.7):
    # change label_by to ['2', '4', '5', '6', '8'] if you want to label by position?
    # if experiment is run across domains (this function has to be iterated for each position
    train_split, test_split = [], []

    for grasp_lbl in label_by:

        temp_list = get_grasp_list(dir_list, grasp_lbl)
        split_size = int(np.floor(split_ratio * len(temp_list)))
        train_samples = np.random.choice(temp_list, size=split_size, replace=False)
        test_samples = [elem for elem in temp_list if elem not in train_samples]
        train_split.append(train_samples)
        test_split.append(test_samples)

    return flatten(train_split),  flatten(test_split)



def get_trial_data_split(trial_blocks, trial_grasps, train_trials_no=3):
    train_trials, test_trials = [], []
    for lbl in set(trial_grasps):
        temp_idx = np.argwhere(lbl == trial_grasps).flatten()

        train_temp = np.random.choice(temp_idx, size=train_trials_no, replace=False)
        test_temp = [elem for elem in temp_idx if elem not in train_temp]
        print('temp idx', temp_idx, 'train', train_temp)

        # Get corresponding trial block indices (from 0-149)
        get_vals1 = itemgetter(*train_temp)
        get_vals2 = itemgetter(*test_temp)
        train_trials.extend(get_vals1(trial_blocks))
        test_trials.extend(get_vals2(trial_blocks))
    print(train_trials, test_trials)
    return train_trials, test_trials

def merge_raw_data(trial_keys, data):
    merged_data = None
    for key in trial_keys:
        if merged_data is None:
            merged_data = data[key][:]
        else:
            merged_data = np.hstack((merged_data, data[key][:]))
    return merged_data

def norm_data(data):
    # scaler = StandardScaler()
    # scaler.fit(data, axis=1)
    # return scaler.mean_, scaler.var_
    return np.mean(data, axis=1), np.std(data, axis=1)

