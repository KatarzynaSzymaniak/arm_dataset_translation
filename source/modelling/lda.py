import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from natsort import natsorted
from source.data.read_raw import get_files_list_NEW
from source.data.rw_processed import merge_data, get_files_list, get_grasp_list
from sklearn.preprocessing import StandardScaler
# from source.data.rw_processed import get_data
from source.data.data_split import get_data_split
# def get_files_list(args, positions, file_extension='.npy', merged_files_only=False, lbl_simple_count=False):
#     '''
#     :param args:  each directory in args corresponds to a new grip
#     :param lbl_indicator:
#     :return:
#     '''
#     labels = 0
#     files = []
#     for dirr in args:
#         dirr_positions = [os.path.join(dirr, 'position_{}'.format(i)) for i in positions]
#         data_sub = [[os.path.join(dir_sub, elem) for elem in os.listdir(dir_sub) if elem[-4:] == file_extension]
#                     for dir_sub in dirr_positions]
#         data_sub = natsorted(flatten(data_sub))  # make sure if only one position taken, the array is flatten
#         #print('data sub', data_sub)
#         if merged_files_only is True:
#             lbl_elem = np.array([elem for elem in data_sub if elem.split('\\')[-1] == str('merged.npy')])
#         else:
#             lbl_elem = np.array([elem for elem in data_sub if (elem.split('\\')[-1] != str('labels.npy')
#                                                                and elem.split('\\')[-1] != str('merged.npy'))])
#
#         files.append(lbl_elem)
#         labels += len(lbl_elem)
#     return labels, flatten(files)


#
# def flatten(xss):
#     return [x for xs in xss for x in xs]


# def get_position_lbl(files_list):
#     # this function has to be customized for this work.
#     arr = []
#     for i in range(len(files_list)):
#         temp = int(files_list[i].split('\\')[-2].split('_')[1])
#         #print('temp', temp)
#         arr.append(temp)
#     return arr


# def write_all_data_paths(args, positions, files_extension='.npy', file_path=None, lbl_rec=False, merged_files_only=False):
#     '''
#
#     :param args: Has to be a list
#     :param positions: list position in  a 3 by 3 grid
#     :param files_extension: string
#     :param file_path:
#     :return:
#     '''
#     labels, files_list = get_files_list(args, positions, file_extension=files_extension,merged_files_only=merged_files_only)
#
#     if lbl_rec is True:
#         arr = get_position_lbl(files_list)
#         files_list = np.hstack((np.array(files_list).reshape(-1,1), np.array(arr).reshape(-1,1)))
#
#     # files_list = np.hstack((np.array(files_list).reshape(-1,1), np.array(labels).reshape(-1,1)))
#     # print('files_list', files_list)
#
#     if file_path is not None:
#         pd.DataFrame(files_list).to_csv(file_path, header=False, index=False)
#
#     return files_list  # flatten(files_list)

# def merge_data(file_list):
#     # Initialize an empty list to store the arrays
#     arrays, lbls = [], []
#
#     # Load the data from each file and append to the list
#     for file in file_list:
#         data = np.load(file)
#         print(f'single file data: {file}, {data.shape}')
#         arrays.append(data)
#
#         grasp = file.split('\\')[-1].split('_')[-2]
#         temp_lbl = np.repeat(int(grasp), data.shape[0]).reshape(-1,1)
#         lbls.append(temp_lbl)
#
#     # Merge the arrays into a single array along the row dimension
#     merged_array = np.vstack(arrays)
#     merged_lbls = np.vstack(lbls)
#
#     # Print the shape of the merged array
#     print('merged_array.shape', merged_array.shape)
#     return merged_array, merged_lbls

# def get_grasp_flist(dir_list, grasp_lbl):
#     filtered_files = []
#
#     # Iterate through the original list
#     for file in dir_list:
#         # Extract the last part after splitting by '_'
#         file_lbl = file.split('\\')[-1].split('_')[-2]
#         if file_lbl == grasp_lbl:
#             filtered_files.append(file)
#
#     return filtered_files
#
# def get_lbls(files):
#     lbls = []
#     for file in files:
#         temp_lbl = file.split('\\')[-1].split('_')[-2]
#         lbls.append(temp_lbl)
#     return flatten(lbls)
from source.data.util import flatten

def lda_acc(x_test, y_test, x_train, y_train, positions):
    scaler = StandardScaler()
    scaler = scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # Create an LDA classifier
    lda = LDA()
    lda.fit(x_train, y_train.ravel())
    y_pred = lda.predict(x_test)
    accuracy = accuracy_score(y_test.ravel(), y_pred)
    #print(f"Grasp: {positions}, Accuracy:, {accuracy}")
    return accuracy


def remove_element_by_index(lst, index):
    return lst[:index] + lst[index+1:]


def run(_dir, grasp_config, classes):

    for i in classes:
        x_test, y_test, x_train, y_train = None, None, None, None
        for positions in grasp_config:
            labels, files = get_files_list([_dir], positions,
                                           file_extension='.npy', merged_files_only=False,
                                           lbl_simple_count=False)

            # from that position, extract only files with labels corresponding to the class of interest

            _files = [file for file in files if file.split('\\')[-1].split('_')[-2]==str(i)]
            # print('Position', _files[0].split('\\')[-1].split('_')[-1][:-4])


            train_split, test_split = get_data_split(_files)
            if x_test is None:
                x_test, y_test = merge_data(test_split, if_position=True)
                x_train, y_train = merge_data(train_split, if_position=True)
            else:
                x_test_temp, y_test_temp = merge_data(test_split, if_position=True)
                x_train_temp, y_train_temp = merge_data(train_split, if_position=True)
                x_test = np.vstack((x_test, x_test_temp))
                y_test = np.vstack((y_test, y_test_temp))
                x_train = np.vstack((x_train, x_train_temp))
                y_train = np.vstack((y_train, y_train_temp))


            # print('x_test', x_test[0,:])

        accuracy = lda_acc(x_test, y_test, x_train, y_train, f'classify position for class {i}')



def main():
    # ==============================================================================
    # Experiment 1: train and test on the same position across all the 6 classes.
    _dir = r'C:\Users\44784\PycharmProjects\arm_translation_dataset\processed_data\hudgins256_100\participant_5\block1'
    grasp_config = ['2', '4', '5', '6', '8']

    for positions in grasp_config:
        labels, files = get_files_list([_dir], positions,
                                       file_extension='.npy', merged_files_only=False,
                                       lbl_simple_count=False)

        train_split, test_split = get_data_split(files)  # , grasp_config=grasp_config)
        x_test, y_test = merge_data(test_split)
        x_train, y_train = merge_data(train_split)
        accuracy = lda_acc(x_test, y_test, x_train, y_train, f'position {positions}')

    # # ==============================================================================
    # # Experiment 2:  do train and test data split on all the positions
    # _dir = r'C:\Users\44784\PycharmProjects\DA_featinv_paper\processed_data\hudgins256_100\participant_5\block2'
    # grasp_config = ['2', '4', '5', '6', '8']
    #
    # x_test, y_test, x_train, y_train = None, None, None, None
    # for positions in grasp_config:
    #     labels, files = get_files_list([_dir], positions,
    #                                    file_extension='.npy', merged_files_only=False,
    #                                    lbl_simple_count=False)
    #
    #     train_split, test_split = get_data_split(files)#, grasp_config=grasp_config)
    #     if x_test is None: # y_test, x_train, y_train
    #         x_test, y_test = merge_data(test_split)
    #         x_train, y_train = merge_data(train_split)
    #     else:
    #         x_test_temp, y_test_temp = merge_data(test_split)
    #         x_train_temp, y_train_temp = merge_data(train_split)
    #         x_test = np.vstack((x_test, x_test_temp))
    #         y_test = np.vstack((y_test, y_test_temp))
    #         x_train = np.vstack((x_train, x_train_temp))
    #         y_train = np.vstack((y_train, y_train_temp))
    # accuracy = lda_acc(x_test, y_test, x_train, y_train, ' all positions merged for train and test split')
    #
    #
    # # =============================================================================
    # # Experiment 3
    # # # train data on one position and test on the remainings;
    # # e.g.: train on position 2 (all grasps) and test on [4,5,6,8] positions (all classes/grasps)
    # # Train on 1 position and then test on teh remaining ones
    # _dir = r'C:\Users\44784\PycharmProjects\DA_featinv_paper\processed_data\hudgins256_100\participant_5\block2'
    # grasp_config = ['2', '4', '5', '6', '8']
    #
    # for i in range(5):
    #     x_test, y_test, x_train, y_train = None, None, None, None
    #     for positions in grasp_config:
    #         labels, files = get_files_list([_dir], positions,
    #                                        file_extension='.npy', merged_files_only=False,
    #                                        lbl_simple_count=False)
    #
    #         if positions == grasp_config[i]:
    #             x_train, y_train = merge_data(files)
    #         else:
    #             # train_split, test_split = data_split(files, grasp_config=grasp_config)
    #             if x_test is None:
    #                 x_test, y_test = merge_data(files)
    #             else:
    #                 x_test_temp, y_test_temp = merge_data(files)
    #                 x_test = np.vstack((x_test, x_test_temp))
    #                 y_test = np.vstack((y_test, y_test_temp))
    #     accuracy = lda_acc(x_test, y_test, x_train, y_train, f'trained on grasp {grasp_config[i]}, tested on the {remove_element_by_index(grasp_config, i)}')
    #
    #
    # # =============================================================================
    # # Experiment 4
    # # # for each class, classify the position of the data, based on the domains.
    # _dir = r'C:\Users\44784\PycharmProjects\DA_featinv_paper\processed_data\hudgins256_100\participant_5\block2'
    # grasp_config = ['2', '4', '5', '6', '8']
    # classes = [1, 2, 3, 4, 5]  # why havent I included class 6? - supposed ot be rest
    #
    # run(_dir, grasp_config, classes)


    # =============================================================================
    # Experiment 5
    # for each class, classify the position of the data, based on the domains.
    # _dir = r'C:\Users\44784\PycharmProjects\DA_featinv_paper\processed_data\hudgins256_100\participant_5\block2'
    # grasp_config = ['2', '4', '5', '6', '8']
    # classes = [1, 2, 3, 4, 5]
    #
    #
    # for i in classes:
    #     for j in range(len(grasp_config)):
    #         x_test, y_test, x_train, y_train = None, None, None, None
    #
    #         for positions in grasp_config:
    #             labels, files = get_files_list([_dir], positions,
    #                                            file_extension='.npy', merged_files_only=False,
    #                                            lbl_simple_count=False)
    #
    #             # from that position, extract only files with labels corresponding to the class of interest
    #
    #             _files = [file for file in files if file.split('\\')[-1].split('_')[-2] == str(i)]
    #             print('Position', _files[0].split('\\')[-1].split('_')[-1][:-4])
    #
    #             if positions == grasp_config[j]:
    #                 x_train, y_train = merge_data(_files, if_position=True)
    #             else:
    #                 if x_test is None:
    #                     x_test, y_test = merge_data(_files, if_position=True)
    #                 else:
    #                     x_test_temp, y_test_temp = merge_data(_files, if_position=True)
    #                     x_test = np.vstack((x_test, x_test_temp))
    #                     y_test = np.vstack((y_test, y_test_temp))
    #
    #         accuracy = lda_acc(x_test, y_test, x_train, y_train,
    #                            f'classify position for class {i}, trained on grasp {grasp_config[j]},'
    #                            f'tested on the {remove_element_by_index(grasp_config, i)}')


#
    # # NEW VERSION ONLY FOR THE LDA WITH POS2_2 WHERE DTA IS SPLIT BY TRIALS BEFORE PROCESSING
    #
    # _dir = r'C:\Users\44784\PycharmProjects\DA_featinv_paper\processed_data\hudgins256_100\participant_5\block1\pos2_2\train'
    # files_train = get_files_list_NEW([_dir],
    #                                file_extension='.npy', merged_files_only=False,
    #                                lbl_simple_count=False)
    # x_train, y_train = merge_data(files_train)
    #
    # _dir_test = r'C:\Users\44784\PycharmProjects\DA_featinv_paper\processed_data\hudgins256_100\participant_5\block1\pos2_2\test'
    # files_test = get_files_list_NEW([_dir_test],
    #                                  file_extension='.npy', merged_files_only=False,
    #                                  lbl_simple_count=False)
    # x_test, y_test = merge_data(files_test)
    #
    # accuracy = lda_acc(x_test, y_test, x_train, y_train, '2')
    #

    # ======================== Processed Data from experiment (when recorded; Iris) ============================
    # p1_label = r'C:\Users\44784\PycharmProjects\DA_featinv_paper\data\participant_5\participant_5_day1_block1_20230113160306\trials.csv'
    # p1 = r"C:\Users\44784\PycharmProjects\DA_featinv_paper\data\participant_5\participant_5_day1_block1_20230113160306\emg_proc.hdf5"
    # trial_keys, data = get_data(p1)
    # print(trial_keys)
    # for elem in trial_keys:
    #     print('data', data[elem])
    #
    # print(data['0'][:])


if __name__ == "__main__":
    main()
