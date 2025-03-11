"""Dataset loading patient images.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self,
                 path_data: str,
                 nb_slices_volume: int = 128,
                 id_patients: list[int] = None):
        
        self.__path_data = path_data
        self.__nb_slices_volume = nb_slices_volume
        self.__id_patients = id_patients

    def __len__(self):
        if self.__id_patients is not None:
            length = len(self.__id_patients)
        else:
            length = len(os.listdir(self.__path_data))
        return length

    def __getitem__(self, idx):

        id_patient = idx
        if self.__id_patients is not None:
            id_patient = self.__id_patients[idx]

        # load images
        filename_data = os.path.join(self.__path_data, f"{id_patient}.npy")

        image = np.load(filename_data)

        # trim volume by bottom of body is requested
        if self.__nb_slices_volume is not None:
            image = image[:, -self.__nb_slices_volume:]

        processed_image = torch.tensor(image, dtype=torch.float32)

        return processed_image
