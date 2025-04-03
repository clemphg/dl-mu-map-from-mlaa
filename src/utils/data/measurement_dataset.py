"""Dataset loading patient images.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MeasurementDataset(Dataset):
    def __init__(self,
                 path_data: str,
                 id_patients: list[int] = None):
        
        self.__path_data = path_data
        self.id_patients = id_patients

        list_files_data = os.listdir(path_data)
        if id_patients is not None:
            id_patients_str = [str(id) for id in id_patients]
            self.__filenames_data = [file for file in list_files_data if file.split('.')[0].split('_')[0] in id_patients_str]
        else:
            self.__filenames_data = list_files_data

    def __len__(self):
        return len(self.__filenames_data)

    def __getitem__(self, idx):
        filename = self.__filenames_data[idx]

        # load images
        filename_data = os.path.join(self.__path_data, filename)

        ybar = np.load(filename_data)

        processed_ybar = torch.tensor(ybar, dtype=torch.float32)

        return filename, processed_ybar
