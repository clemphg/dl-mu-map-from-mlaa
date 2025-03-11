"""Compute mean of attenuation in dataset.
Used as initialization in MLAA.
"""

import os
import sys
from tqdm import tqdm
import numpy as np
import pickle as pkl
import dotenv

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

dotenv.load_dotenv()

from src.utils.data.image_dataset import ImageDataset

def save_mean(mean, file_path):
    with open(file_path, 'wb') as f:
        pkl.dump(mean, f)

def main():
    out_filename = os.environ['PATH_MEAN_ATN']
    path_data = os.environ['PATH_REFERENCE']
    path_id_patients = os.environ['PATH_ID_PATIENTS']

    with open(path_id_patients, 'rb') as f:
        id_patients = pkl.load(f)
        
    id_patients_train = id_patients['train']

    dataset = ImageDataset(path_data=path_data,
                           nb_slices_volume=64,
                           id_patients=id_patients_train)
    
    running_meansum = 0

    for idx in tqdm(range(len(dataset)), ncols=100):
        filename, mu_img = dataset[idx][1]
        running_meansum += torch.mean(mu_img)

    # save mean
    mean = running_meansum/len(dataset)
    save_mean({'mean': mean}, out_filename)


if __name__=="__main__":
    main()