
import os
import numpy as np
from source.data.windower import Windower
from source.data.dataIO import DataReadWrite as dataIO
from source.data.read_raw import get_data, get_trial_pos_grasp

NO_TRIALS = 5
GLOBAL_REC_TRIALS = 150

# TODO:  yaml with config
def get_config():
    block = 1
    participant = 1
    grasps = ['rest', 'tripod', 'lateral', 'pointer', 'power']
    block_configuration = {'+': [2, 4, 5, 6, 8], 'x': [1, 3, 5, 7, 9]}
    trials = 5
    return {'block': block,
            'participant': participant,
            'grips': grasps,
            'block_config': block_configuration,
            'trials': trials
            }



def run_trial_preprocess_NEW(trials, data_dict, y_dict, mean, std,  if_norm=False,
                         _dir= r'C:\Users\44784\PycharmProjects\DA_featinv_paper\processed_data\hudgins256_100\participant_5\block1',
                         extract_func = "hudgins_feats"):
    '''
    this function is an updated version of run_trial_preprocess,
    where we split the trials before processing the dataset

    -> we need to pass std and mean for both training and testing in this case
    :param trials_no:
    :param data_dict:
    :param y_dict:
    :param _dir:
    :return:
    '''

    for elem in trials:
        temp_x = data_dict[str(elem)][:]
        temp_y = np.zeros(temp_x.shape[1])
        temp_y[:] = y_dict[elem]['grasp']

        # pos = 'position_' + str(y_dict[elem]['position'])
        temp_dir = _dir  # os.path.join(_dir, pos)
        if dataIO().check_dir_exist(temp_dir) is False:
            os.mkdir(temp_dir)

        slide = Windower(mean=mean, std=std)
        slide.slide_window(temp_x, temp_y, extract_func, trial_key=elem, position=y_dict[elem]['position'],
                           segment_size=256, step=100, rootdir=temp_dir, if_offset_corr=False)  # if_norm=False



def run_trial_preprocess(trials_no, data_dict, y_dict,
                         _dir= r'C:\Users\44784\PycharmProjects\DA_featinv_paper\processed_data\hudgins256_100\participant_5\block1'):
    for i in range(trials_no):
        temp_x = data_dict[str(i)][:]
        temp_y = np.zeros(temp_x.shape[1])
        temp_y[:] = y_dict[i]['grasp']

        pos = 'position_' + str(y_dict[i]['position'])
        temp_dir = os.path.join(_dir, pos)
        if dataIO().check_dir_exist(temp_dir) is False:
            os.mkdir(temp_dir)

        slide = Windower()
        slide.slide_window(temp_x, temp_y, "hudgins_feats", trial_key=i, position=y_dict[i]['position'],
                segment_size=256, step=100, rootdir=temp_dir, if_offset_corr=True)



def main():
    cwd = os.getcwd()
    components = cwd.split(os.sep)[:-2]
    project_path = cwd #.sep.join(components)
    print('project path', cwd)
    for i in range(1,11): #sub
        #if i < 9:
        #    continue
        for day in range(1,3): # 2 days
            #if day == 2:
            #    continue
            for block in range(1,3): # 2 blocks
                sub_prime = 'participant_' + str(i)
                sub = 'participant' + str(i)+'_'
                d = 'day'+str(day) + '_block' + str(block)

                read_path = os.path.join(r"data", sub_prime, str(sub+d))
                print('rr', read_path)

                # 1. Load data - one subject
                p1 = os.path.join(project_path, read_path, "emg_raw.hdf5")
                p1_label = os.path.join(project_path, read_path, "trials.csv")
                trials_keys, data = get_data(p1)
                lbls_dict = get_trial_pos_grasp(p1_label)

                # 2. Slide window and process the data.
                save_dir = os.path.join(project_path, "processed_data","hudgins256_100", sub_prime, d)
                print('save dir:', save_dir)
                trials_no = 150  # lbls_dict.keys().__len__()-> throws error for 151 trials (last trial being empty)
                run_trial_preprocess(trials_no, data, lbls_dict, _dir=save_dir)

    # ----- RUN SEPARATELY ------------
    # # 1. Load data - one subject
    # p1 = os.path.join(project_path, r"C:\Users\44784\PycharmProjects\arm_translation_dataset\data\participant_1\participant1_day1_block1\emg_raw.hdf5")
    # p1_label = os.path.join(project_path, r"C:\Users\44784\PycharmProjects\arm_translation_dataset\data\participant_1\participant1_day1_block1\trials.csv")
    # trials_keys, data = get_data(p1)
    # print('trials_keys', trials_keys)
    # lbls_dict = get_trial_pos_grasp(p1_label)
    #
    # # 2. Slide window and process the data.
    # save_dir = os.path.join(project_path, "processed_data\hudgins256_100\participant_2\participant2_day1_block1" )
    # print('save dir:', save_dir)
    # trials_no = 150  # lbls_dict.keys().__len__()-> throws error for 151 trials (last trial being empty)
    # # run_trial_preprocess(trials_no, data, lbls_dict, _dir=save_dir)

if __name__ == "__main__":
    main()
