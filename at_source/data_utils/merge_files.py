

# this fixed no module named.... error
# set -Ux PYTHONPATH $PYTHONPATH /home/kasia/arm_dataset_translation/

import os
import torch
import numpy as np
from at_source.configs.utils import get_configs
import torch

# args = get_configs(os.path.join(os.getcwd(), 'AT_Great', 'experiment_configs', 'data_configs', 'great.yaml'))


# # scrape files from the folder for a single subejct
# def scrape_files(path, pattern):
#     # Use to find the fiels from a predefined position and list them
#     return [f for f in os.listdir(path) if fnmatch.fnmatch(f, f'*{pattern}*')]
# # find files
# def find(pattern, path):
#     # Use to find the fiels from a predefined position and list them
#     return [f for f in os.listdir(path) if fnmatch.fnmatch(f, f'*{pattern}*')]



var_pos = {
  'day1_block1': [2,4,5,6,8],
  'day1_block2': [2,4,5,6,8],
  'day2_block1': [1,3,5,7,9],
  'day2_block2': [1,3,5,7,9]}

def list_subdirs(subs_path):
    return [os.path.join(subs_path, d) for d in os.listdir(subs_path) if os.path.isdir(os.path.join(subs_path, d))]


def merge_pos_data(path, sub_save_dir, sub_name):
    pos_list = var_pos[os.path.basename(path)]
    print('pos_list: ', pos_list)
    positions = list_subdirs(path)
    block = os.path.basename(path)
    for poss in positions:
        get_pos_data(poss,  sub_save_dir, block)


# save position files to a single tensor
def get_pos_data(path, sub_save_dir, block):
    # Load the numpy files and convert them to tensors
    file_list = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.npy')]
    
    tensors = [torch.from_numpy(np.load(os.path.join(path, file))) for file in os.listdir(path) if file.endswith('.npy')]
    # Stack the tensors along a new dimension
    stacked_tensors = torch.stack(tensors)
    save_to = os.path.join(sub_save_dir, block+'_'+os.path.basename(path)+'.pt')
    torch.save(stacked_tensors, save_to)

    # save the lbl of the data 
    get_pos_lbl_data(file_list, sub_save_dir, block, path)


def get_pos_lbl_data(file_names, sub_save_dir, block, pos):

    # print(os.path.splitext(file_names[0]), os.path.splitext(file_names[0])[1].split('_')[:], os.path.split(os.path.splitext(file_names[0])[0])[1].split('_')[:])
    # Split each file name, remove the extension, take the last two parts, and convert to tensor
    # tensors = [torch.tensor([int(part) for part in os.path.splitext(file_name)[0].split('_')[-2:]]) for file_name in file_names]
    tensors = [torch.tensor([int(part) for part in os.path.split(os.path.splitext(file_name)[0])[1].split('_')[:]]) for file_name in file_names]
    stacked_tensors = torch.stack(tensors)
    stacked_tensors[:, [0,2,3]]
    save_to = os.path.join(sub_save_dir, block+'_'+os.path.basename(pos)+'_y.pt')
    torch.save(stacked_tensors, save_to)



    

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)    


def main():
    root_dir = os.getcwd() #os.path.split(os.getcwd())[0]
    print(root_dir, 'root_dir')
    subs_path = os.path.join(root_dir,'processed_data','hudgins256_100')
    subs = list_subdirs(subs_path)
    print('subs: ', subs)

    for s in subs:
        sub_name = os.path.basename(s)
        sub_save_dir = os.path.join(root_dir, 'at_source', 'processed_data', sub_name)
        check_dir(sub_save_dir)

        block_dirlist = list_subdirs(s)
        print('block_dirlist: ', block_dirlist)
        for block in block_dirlist:
            merge_pos_data(block, sub_save_dir, sub_name)


    
    
if __name__ == '__main__':
    main()

