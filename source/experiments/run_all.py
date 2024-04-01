import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from natsort import natsorted
from source.data.rw_processed import get_files_list, get_grasp_list
from source.data.read_raw import get_files_list_NEW
from sklearn.preprocessing import StandardScaler
# from source.data.rw_data import get_data
from source.modelling.lda import lda_acc, remove_element_by_index
from source.data.rw_processed import merge_data
from source.data.util import flatten
import yaml
import random
import csv
import timeit
from source.data.read_raw import get_subdirs
from source.data.dataIO import DataReadWrite as dataIO

def get_configparser():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config


def run():
    # do cross validation and save all the results

    # do set of different split for train and test. save the seed and info to replicate the
    # results afterwards.
    # save all acc
    acc = None
    return acc

positions = [2,4,5,6,8]

def run_naive_da():
    pass

def run_naive_da_onevsall(grasp_config=['2','4','5','6','8'],
                 _dir=r'C:\Users\44784\PycharmProjects\DA_featinv_paper\processed_data\hudgins256_100\participant_5\block2'
                 ):

    for i in range(5):
        x_test, y_test, x_train, y_train = None, None, None, None
        for positions in grasp_config:
            labels, files = get_files_list([_dir], positions,
                                           file_extension='.npy', merged_files_only=False,
                                           lbl_simple_count=False)

            if positions == grasp_config[i]:
                x_train, y_train = merge_data(files)
            else:
                # train_split, test_split = data_split(files, grasp_config=grasp_config)
                if x_test is None:
                    x_test, y_test = merge_data(files)
                else:
                    x_test_temp, y_test_temp = merge_data(files)
                    x_test = np.vstack((x_test, x_test_temp))
                    y_test = np.vstack((y_test, y_test_temp))
        accuracy = lda_acc(x_test, y_test, x_train, y_train,
                           f'trained on grasp {grasp_config[i]}, tested on the {remove_element_by_index(grasp_config, i)}')

    # need to save results

#TODO remove this function
def get_config_run(classif_mode):
    if classif_mode == 'da_onevsall':
        run_naive_da_onevsall()
    if classif_mode == 'da_one_to_one':
        pass
    if classif_mode == 'source_only':
        pass
    if classif_mode == 'position':
        pass
    if classif_mode == 'all':
        pass

#TODO remove this function
def get_subject_paths(root_dir):
    paths = None
    return paths


def get_grasp_trials(files, classes=['1','2','3','4','5','6']):
    '''
    create dictionary of trials for each class and
    shuffle them to randomise
    :param files:
    :param classes:
    :return:
    '''
    # get grasps (lbl) trials only from on position
    temp_files = [os.path.split(files[i])[1] for i in range(len(files))]
    _dict = {}
    for grasp in classes:
        grasp_files = [[temp_files[i], int(temp_files[i].split('_')[0])]
                       for i in range(len(files)) if temp_files[i].split('_')[-2] == grasp]
        # print(f'grasp_files {grasp_files}')
        grasp_trials, _, _ = np.unique(np.array(grasp_files)[:, 1], return_index=True, return_counts=True)

        # shuffle trials to ensure that there is no bias in cross validation later on!
        grasp_trials = list(grasp_trials)
        shuffled = random.sample(grasp_trials, len(grasp_trials))
        # todo cast the type:
        shuffled = list(map(int, shuffled))
        _dict.update({grasp: shuffled})
    return _dict

def split(_dict):
    '''
    Split trials, each key in the dictionary represents a class; value: trials within this class.
    Split data using a k-1 approach.
    :param _dict:
    :return:
    '''
    temp_arr = []
    for key, value in _dict.items():
        # key is a class, value is all trials
        for i in range(len(value)):
            temp = value[:i] + value[i+1:]
            if key == '1': # if len(temp_arr) == 0:  #
                temp_arr.append(temp)
            else:
                temp_arr[i].extend(temp)

            # TODO simplify?
            # temp_arr[i] = temp

    # print('temp arr', np.array(temp_arr))
    return np.array(temp_arr)



def get_trials_files(trials, file_list):
    _list = []
    for t in trials:
        for file in file_list:
            #print(f' file{file}, t {t}')
            if file.split('\\')[-1].split('_')[0] == t:
                _list.append(file)

    return _list


def run_all_combinations(positions, data_block,
                         _dir=r'C:\Users\44784\PycharmProjects\DA_featinv_paper\
                         processed_data\hudgins256_100\participant_5\block2',
                         if_cross_val=False, save_f=None):
    _arr = None
    # do cross validation on each subject - get acc and std based on that
    for source in positions:
        for target in positions:
            #if source != target:
            print(f'____________________________ Source {source}, target {target}____________________________')
            s_labels, s_files = get_files_list([_dir], source,
                                           file_extension='.npy', merged_files_only=False,
                                           lbl_simple_count=False)
            t_labels, t_files = get_files_list([_dir], target,
                                           file_extension='.npy', merged_files_only=False,
                                           lbl_simple_count=False)

            if if_cross_val:
                # split data into n groups for cv validation
                _dict = get_grasp_trials(s_files)
                train_trials_ids = split(_dict)
                save_res = []
                for i in range(train_trials_ids.shape[0]):
                    temp_train_list = get_trials_files(train_trials_ids[i], s_files)
                    if source == target:
                        t_files = [elem for elem in s_files if elem not in temp_train_list]
                        s, t, temp_acc = run_whole_train(source, target, temp_train_list, t_files, i=i)
                    else:
                        # print(f'S:{source}, T: {target}, split: {i}, temp list: {len(temp_train_list)}')
                        s, t, temp_acc = run_whole_train(source, target, temp_train_list, t_files, i=i)
                    # save_res.append(temp_acc)
                    if _arr is None:
                        _arr = np.array(([int(source), int(target), temp_acc]))
                    else:
                        _arr = np.vstack((_arr, [int(source), int(target), temp_acc]))

                # print(f' save_res, {save_res}')
                # print(f'Average: {np.mean(np.array(_arr[:,2]))}')

            else:
                temp_acc = run_whole_train(source, target, s_files, t_files)

    if save_f is not None:
        if dataIO().check_dir_exist(save_f) is False:
            os.mkdir(save_f)
        file_path = os.path.join(save_f, str(data_block) + '.csv')
        print('_arr[:,0], _arr[:,1], _arr[:,2]', _arr[:,0], _arr[:,1], _arr[:,2])
        save_arrays_to_csv(file_path, _arr[:,0], _arr[:,1], _arr[:,2])
    # x_train, y_train = merge_data(s_files)
    # x_test, y_test = merge_data(t_files)
    # accuracy = lda_acc(x_test, y_test, x_train, y_train, f'position {positions}')
    # print(f' Source {source}, Target {target}: Acc: {accuracy}')
    # _arr.append([source, target, accuracy])
    # # save the stats
    # print('arr', _arr)



def run_whole_train(source, target, s_files, t_files, i=None):
    x_train, y_train = merge_data(s_files)
    x_test, y_test = merge_data(t_files)
    accuracy = lda_acc(x_test, y_test, x_train, y_train, f'position {positions}')
    # print(f' Source {source}, Target {target}: Acc: {accuracy}, split: {i}')
    return int(source), int(target), accuracy

def save_arrays_to_csv(file_path, sources, targets, accuracies):
    # Combine the arrays into a list of rows
    rows = list(zip(sources, targets, accuracies))

    # Open the CSV file in write mode
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the header row
        writer.writerow(['Source', 'Target', 'Accuracy'])

        # Write the data rows
        writer.writerows(rows)

def read_arrays_from_csv(file_path):
    sources = []
    targets = []
    accuracies = []

    # Open the CSV file in read mode
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)

        # Skip the header row
        next(reader)

        # Read the data rows and populate the arrays
        for row in reader:
            sources.append(row[0])
            targets.append(row[1])
            accuracies.append(float(row[2]))

    return sources, targets, accuracies


def avg_across_sub(sources, targets, files:list):
    for source, target in zip(sources, targets):
        temp=[]
        for f in files:
            (s, t, accs) = zip(read_arrays_from_csv(f))
            for ss, tt, acc in (s,t, accs):
                if ss == source and tt == target:
                    temp.append(acc)


def main():
    # run only 1 block! (for now  - see time)
    temp_folders = ['day1_block1', 'day2_block1']
    start = timeit.timeit()
    processed_path = r"C:\Users\44784\PycharmProjects\arm_translation_dataset\processed_data\hudgins256_100"
    subs = get_subdirs(processed_path)
    for sub in subs:

        temp_sub_path = os.path.join(processed_path, sub)
        data_blocks = get_subdirs(temp_sub_path)
        for data_block in data_blocks:
            save_path = os.path.join(r'C:\Users\44784\PycharmProjects\arm_translation_dataset\results', sub)
            # if dataIO().check_dir_exist(save_path) is False:
            #     os.mkdir(save_path)
            # file_path = os.path.join(save_path, str(data_block)+'.csv')
            # print(file_path, 'file_path')
            # save_arrays_to_csv(file_path, [1], [2], [4])
            if data_block == temp_folders[0]:
                positions = ['2', '4', '5', '6', '8']
                _dir = os.path.join(temp_sub_path, data_block)
                run_all_combinations(positions, data_block, _dir=_dir, if_cross_val=True, save_f=save_path)

            elif data_block == temp_folders[1]:
                positions = ['1', '3', '5', '7', '9']
                _dir = os.path.join(temp_sub_path, data_block)
                run_all_combinations(positions, data_block, _dir=_dir, if_cross_val=True, save_f=save_path)




    positions = ['2', '4', '5', '6', '8']
    _dir = r'C:\Users\44784\PycharmProjects\DA_featinv_paper\processed_data\hudgins256_100\participant_5\block2'
    # run_all_combinations(positions, _dir=_dir, if_cross_val=True)

    # config = get_configparser()
    # grasp_config = config['General']['block_configuration']
    # root_path = config['General']['root_path']
    # classes = config['General']['classes']

    # subjects_paths = get_subject_paths(root_path)
    # for sub in subjects_paths:
    #     run_all_combinations(simple_lda.run(sub, grasp_config, classes))


    #####
    # .yaml have all the configs to run
    # lda class    # data split for cross validation - based on trials and all the data
    #

    #

    end = timeit.timeit()
    print(end - start)

if __name__ == "__main__":
    main()
