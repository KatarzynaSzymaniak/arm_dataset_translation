import numpy as np
import torch 
import os
import sys
from sklearn.model_selection import StratifiedKFold

# def cv_split(data, labels, n_splits=5, seed=0):
#     # Extract trial and grasp columns from labels
#     trial = labels[:, 0]
#     grasp = labels[:, 1]

#     # Get unique trials for each grasp
#     unique_trials = {}
#     for g in np.unique(grasp):
#         unique_trials[g] = np.unique(trial[grasp == g])
#         print(' unique_trials, ',  unique_trials)

#     # Shuffle trials for each grasp
#     np.random.seed(seed)
#     for g in unique_trials:
#         np.random.shuffle(unique_trials[g])

#     # Create cross-validation splits
#     splits = []
#     for i in range(n_splits):
#         split = []
#         for g in unique_trials:
#             split.extend(unique_trials[g][i::n_splits])
#         splits.append(split)


def get_trial_grasp_keys(lbl):
    unique_mappings, indices = np.unique(lbl[:, [0, 2]].numpy(), axis=0, return_index=True)
    return unique_mappings


def trials_split(unique_mappings, k_folds=5):
    skf = StratifiedKFold(n_splits=k_folds, random_state=42, shuffle=True) # False
    b = skf.split(unique_mappings[:,0], unique_mappings[:,1])
    fold_dict = {}
    for fold, (train_ids, test_ids) in enumerate(b):
        print(f'FOLD {fold}')
        print('-------------')
        fold_dict[fold] = {'train_ids': train_ids.tolist(), 'test_ids': test_ids.tolist()}
        # print('fold trials', fold_dict[fold])
    return fold_dict


def get_lbl_data(lbl):
    trial = lbl[:,0]  
    # lbl[:, 1] is A COUNTER 
    lbl = lbl[:,2]
    pos = lbl[:,3]
    return trial, lbl, pos

def get_id_data(idx, data, lbl):
    idx = np.array(idx)
    return data[idx], lbl[idx]
 
def split(X):
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

def get_idx_trials(trials, lbl):
    idx = np.where(np.isin(trials, lbl[:, 0]))[0]
    print('idx', idx)
    print(idx.shape)
    return idx

def get_data():
    d1_path = r'/home/kasia/arm_dataset_translation/at_source/processed_data/participant_1/day1_block1_position_2.pt'
    y1_path = r'/home/kasia/arm_dataset_translation/at_source/processed_data/participant_1/day1_block1_position_2_y.pt'
    d2_path = r'/home/kasia/arm_dataset_translation/at_source/processed_data/participant_1/day1_block2_position_2.pt'
    y2_path = r'/home/kasia/arm_dataset_translation/at_source/processed_data/participant_1/day1_block2_position_2_y.pt'
    d1 = torch.load(d1_path)
    y1 = torch.load(y1_path)
    d2 = torch.load(d2_path)
    y2 = torch.load(y2_path)
    return d1, y1, d2, y2

def get_fold_trials(trial_mappings, fold_dict, fold):
    temp_idx = np.array(fold_dict[fold]['train_ids'])
    temp_test_idx = np.array(fold_dict[fold]['test_ids'])
    fold_trials = trial_mappings[temp_idx]
    fold_test_trials = trial_mappings[temp_test_idx]
    return fold_trials, fold_test_trials


def get_fold_data(fold_trials, data, lbl):
    indices = np.where(np.isin(lbl[:, 0], fold_trials[:, 0]))[0]
    return data[indices], lbl[indices]
    
def get_data(d1, y1):
    trial_mappings = get_trial_grasp_keys(y1)
    fold_dict = trials_split(trial_mappings)
    fold_trials, fold_test_trials = get_fold_trials(trial_mappings, fold_dict, fold=0)
    get_fold_data(fold_trials, d1, y1)


def main():
    d1, y1, d2, y2 = get_data()
    print(y1.shape) 

    trial_mappings = get_trial_grasp_keys(y1)
    fold_dict = trials_split(trial_mappings)
    fold_trials, fold_test_trials = get_fold_trials(trial_mappings, fold_dict, fold=0)
    get_fold_data(fold_trials, d1, y1)
    
if __name__ == '__main__':
    main()