

"""Dataset combining inputs lambda-MLAA and mu-MLAA with target mu-CT (+AC PET if joint estimation).

Training is done on 32x32x32 patches. Each time a different patch is selected for a given patient.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class MeasuresImageDataset(Dataset):
    def __init__(self,
                 id_patients: list[int],
                 path_inputs: str,
                 path_measures: str,
                 norm_pet = None,
                 norm_mu = None,
                 return_fn: bool=False):
        
        self.__id_patients = id_patients
        
        self.__path_inputs = path_inputs
        self.__path_measures = path_measures


        id_patients_str = [str(id) for id in id_patients]

        list_files_inputs = os.listdir(path_inputs)
        self.__filenames_inputs = [file for file in list_files_inputs if file.split('.')[0].split('_')[0] in id_patients_str]

        list_files_measures = os.listdir(path_measures)
        self.__filenames_measures = [file for file in list_files_measures if file.split('.')[0].split('_')[0] in id_patients_str]

        self.__filenames_inputs = sorted(self.__filenames_inputs)

        for f in self.__filenames_inputs:
            assert f in self.__filenames_measures

        self.norm_pet = norm_pet
        self.norm_mu = norm_mu

        self.return_fn = return_fn

    def __len__(self):
        return len(self.__filenames_inputs)

    def __getitem__(self, idx):
        filename = self.__filenames_inputs[idx]

        # load images
        filename_input = os.path.join(self.__path_inputs, filename)
        filename_measures = os.path.join(self.__path_measures, filename)

        input_img = np.load(filename_input)
        measures = np.load(filename_measures)

        processed_input = input_img.copy()
        # normalize 
        if self.norm_pet is not None:
            processed_input[0] = self.norm_pet(input_img[0], clip=True)
        if self.norm_mu is not None:
            processed_input[1] = self.norm_mu(input_img[1], clip=True)

        processed_input = torch.tensor(processed_input, dtype=torch.float32)
        processed_measures = torch.tensor(measures, dtype=torch.float32).squeeze()

        if self.return_fn:
            return filename, processed_input, processed_measures
        else:
            return processed_input, processed_measures
    