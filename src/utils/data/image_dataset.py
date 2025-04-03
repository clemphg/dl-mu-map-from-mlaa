"""Dataset loading patient images.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self,
                 path_data: str,
                 nb_slices_volume: int = 64,
                 id_patients: list[int] = None,
                 norm_pet = None,
                 norm_mu = None):
        
        self.__path_data = path_data
        self.__nb_slices_volume = nb_slices_volume
        self.__id_patients = id_patients

        list_files_data = os.listdir(path_data)
        if id_patients is not None:
            id_patients_str = [str(id) for id in id_patients]
            self.__filenames_data = [file for file in list_files_data if file.split('.')[0].split('_')[0] in id_patients_str]
        else:
            self.__filenames_data = list_files_data

        self.norm_pet = norm_pet
        self.norm_mu = norm_mu

    def __len__(self):
        return len(self.__filenames_data)

    def __getitem__(self, idx):
        filename = self.__filenames_data[idx]

        # load images
        filename_data = os.path.join(self.__path_data, filename)

        image = np.load(filename_data)

        # trim volume by bottom of body is requested
        if self.__nb_slices_volume is not None:
            image = image[:, -self.__nb_slices_volume:]

        if self.norm_pet is not None:
            image[0] = self.norm_pet(image[0], clip=True)
        if self.norm_mu is not None:
            image[1] = self.norm_mu(image[1], clip=True)

        processed_image = torch.tensor(image, dtype=torch.float32)

        return filename, processed_image
