import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from natsort import natsorted
from operator import itemgetter
import sys
from source.data.dataIO import DataReadWrite as dataIO
from source.data.util import flatten

NO_TRIALS = 5

def merge_data(file_list, if_position=False):
    # Initialize an empty list to store the arrays
    arrays, lbls = [], []

    # Load the data from each file and append to the list
    for file in file_list:
        data = np.load(file, allow_pickle=True)
        # print(f'single file data: {file}, {data.shape}')
        arrays.append(data.flatten())

        if if_position is True:
            # if want to classify based on the position (as a lbl)
            grasp = int(file.split('\\')[-1].split('_')[-1][:-4]) # assuming its in *.npy file
        else:
            # if want to classify based on the grasp (gesture)
            grasp = int(file.split('\\')[-1].split('_')[-2])
        lbls.append(grasp)
        # temp_lbl = np.repeat(int(grasp), data.shape[0]).reshape(-1,1)
        # lbls.append(temp_lbl)

    # Merge the arrays into a single array along the row dimension
    merged_array = np.vstack(arrays)
    merged_lbls = np.vstack(lbls)

    # Print the shape of the merged array
    #  print('merged_array.shape', merged_array.shape)
    return merged_array, merged_lbls


def load_data_npy(file_list, dir, save_dir=None):
    # Load the data from each .npy file and stack them into a new dimension
    data_list = [np.load(os.path.join(dir, f)) for f in file_list]
    merged_data = np.stack(data_list, axis=-1)

    # Save the merged data to a new .npy file
    if save_dir is not None:
        np.save(os.path.join(save_dir, 'merged_data.npy'), merged_data)

    return merged_data

def make_participants_dirs(process_parent_dir):
    data_parent_dir = r"/data"
    participants = dataIO().get_subdirs(data_parent_dir)
    blocks = ['block1', 'block2']

    for i in range(len(participants)):
        for j in range(len(blocks)):
            _path = os.path.join(process_parent_dir, participants[i], blocks[j])
            # check_dir(_path)
            dataIO().check_dir_exist(_path)


def save_processed_data_info(position, mean, std, train_trials, test_trials, fname):
    # save mean, std
    # train test split (based on trials) for that position
    with h5py.File(fname, 'w') as f:
        dset1 = f.create_dataset('Position', data=position)
        dset2 = f.create_dataset('Mean', data=mean)
        dset3 = f.create_dataset('Std', data=std)
        dset1 = f.create_dataset('TrainTrials', data=train_trials)
        dset2 = f.create_dataset('TestTrials', data=test_trials)


def get_files_list(args, positions, file_extension='.npy', merged_files_only=False, lbl_simple_count=False):
    '''
    :param args:  each directory in args corresponds to a new grip
    :param lbl_indicator:
    :return:
    '''
    labels = 0
    files = []
    for dirr in args:
        dirr_positions = [os.path.join(dirr, 'position_{}'.format(i)) for i in positions]
        data_sub = [[os.path.join(dir_sub, elem) for elem in os.listdir(dir_sub) if elem[-4:] == file_extension]
                    for dir_sub in dirr_positions]
        data_sub = natsorted(flatten(data_sub))  # make sure if only one position taken, the array is flatten
        # print('data sub', data_sub)
        if merged_files_only is True:
            lbl_elem = np.array([elem for elem in data_sub if elem.split('\\')[-1] == str('merged.npy')])
        else:
            lbl_elem = np.array([elem for elem in data_sub if (elem.split('\\')[-1] != str('labels.npy')
                                                               and elem.split('\\')[-1] != str('merged.npy'))])

        files.append(lbl_elem)
        labels += len(lbl_elem)
    return labels, flatten(files)


def write_all_data_paths(args, positions, files_extension='.npy', file_path=None, lbl_rec=False, merged_files_only=False):
    '''

    :param args: Has to be a list
    :param positions: list position in  a 3 by 3 grid
    :param files_extension: string
    :param file_path:
    :return:
    '''
    labels, files_list = get_files_list(args, positions, file_extension=files_extension, merged_files_only=merged_files_only)

    if lbl_rec is True:
        arr = get_position_lbl(files_list)
        files_list = np.hstack((np.array(files_list).reshape(-1,1), np.array(arr).reshape(-1,1)))

    # files_list = np.hstack((np.array(files_list).reshape(-1,1), np.array(labels).reshape(-1,1)))
    # print('files_list', files_list)

    if file_path is not None:
        pd.DataFrame(files_list).to_csv(file_path, header=False, index=False)

    return files_list  # flatten(files_list)



def get_position_lbl(files_list):
    # this function has to be customized for this work.
    arr = []
    for i in range(len(files_list)):
        temp = int(files_list[i].split('\\')[-2].split('_')[1])
        arr.append(temp)
    return arr


def get_grasp_list(dir_list, grasp_lbl):
    filtered_files = []

    # Iterate through the original list
    for file in dir_list:
        # Extract the last part after splitting by '_'
        file_lbl = file.split('\\')[-1].split('_')[-2]
        if file_lbl == grasp_lbl:
            filtered_files.append(file)
    return filtered_files


def get_lbls(files):
    lbls = []
    for file in files:
        temp_lbl = file.split('\\')[-1].split('_')[-2]
        lbls.append(temp_lbl)
    return flatten(lbls)