import numpy as np
from source.data.rw_processed import load_data_npy


def get_config():
    pass


def load_data(sub, block, config):
    pass


def get_dataset(sub, block, configuration='x'):
    # load processed data for classification
    if sub == 'participant_1':

        load_data(sub, block, configuration)
    elif sub == 'participant_2':
        load_data(sub, block, configuration)
    elif sub == 'participant_3':
        load_data(sub, block, configuration)
    elif sub == 'participant_4':
        load_data(sub, block, configuration)
    elif sub == 'participant_5':
        load_data(sub, block, configuration)
    elif sub == 'participant_6':
        load_data(sub, block, configuration)
    elif sub == 'participant_7':
        load_data(sub, block, configuration)
    elif sub == 'participant_8':
        load_data(sub, block, configuration)


def get_config_run(classif_mode):
    if classif_mode == 'DA':
        pass
    if classif_mode == 'source_only':
        pass
    if classif_mode == 'position':
        pass
    if classif_mode == 'all':
        pass





def main():
    pass

if __name__ == '__main__':
    main()