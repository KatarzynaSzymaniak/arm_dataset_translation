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


def run_all_combinations(config):
    # do cross validation on each subject - get acc and std based on that
    for source in config:
        for target in config:
            if source != target:

                # run pipeline
                acc = run()

                # save the stats


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

    
def get_config_run(classif_mode):
    if classif_mode == 'DA':
        run_naive_da_onevsall()
    if classif_mode == 'source_only':
        pass
    if classif_mode == 'position':
        pass
    if classif_mode == 'all':
        pass


def get_subject_paths(root_dir):
    paths = None
    return paths



def get_stats():
    # read results and get acc and std from all the runs

    pass



def main():
    config = get_configparser()
    grasp_config = config['General']['block_configuration']
    root_path = config['General']['root_path']
    classes = config['General']['classes']

    # subjects_paths = get_subject_paths(root_path)
    # for sub in subjects_paths:
    #     run_all_combinations(simple_lda.run(sub, grasp_config, classes))


    #####
    # .yaml have all the configs to run
    # lda class
    # data split for cross validation - based on trials and all the data
    #

    #



if __name__ == "__main__":
    main()
