"""Dataset combining inputs lambda-MLAA and mu-MLAA with target mu-CT (+AC PET if joint estimation).

Training is done on 32x32x32 patches. Each time a different patch is selected for a given patient.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self,
                 id_patients: list[int],
                 path_inputs: str,
                 path_targets: str,
                 patch_shape: tuple = (32, 32, 32),
                 nb_slices_volume: int = 128):
        
        self.__id_patients = id_patients
        
        self.__path_inputs = path_inputs
        self.__path_targets = path_targets

        self.__patch_shape = patch_shape

        self.__nb_slices_volume = nb_slices_volume

    def __len__(self):
        return len(self.__id_patients)

    def __getitem__(self, idx):

        id_patient = self.__id_patients[idx]

        # load images
        filename_input = os.path.join(self.__path_inputs, f"{id_patient}.npy")
        filename_target = os.path.join(self.__path_targets, f"{id_patient}.npy")

        input_img = np.load(filename_input)
        target_img = np.load(filename_target)

        # trim volume by bottom of body is requested (mlaa done on sub-volume)
        if self.__nb_slices_volume is not None:
            input_img = input_img[:, -self.__nb_slices_volume:]
            target_img = target_img[:, -self.__nb_slices_volume:]

        # extract 32x32x32 patch whose center is in the body
        body_mask = target_img[1] > 0
        body_indices = np.argwhere(body_mask) # valid indices in the body

        if len(body_indices) == 0:
            # default to the center
            center = np.array(input_img.shape[1:]) // 2
        else:
            center = body_indices[np.random.randint(len(body_indices))]

        processed_input = self.__extract_patch(input_img, center)
        processed_target = self.__extract_patch(target_img, center)

        # normalize
        processed_input = self.__normalize(input_img)
        processed_target = self.__normalize(target_img)

        processed_input = torch.tensor(processed_input, dtype=torch.float32)
        processed_target = torch.tensor(processed_target, dtype=torch.float32)

        return processed_input, processed_target   
    

    def __extract_patch(self, volume, center):
        """Extract a 3D patch centered at the given location."""
        c_x, c_y, c_z = center
        p_x, p_y, p_z = self.__patch_shape
        H, W, D = volume.shape[1:]  # Exclude channel

        # Compute the start and end indices for slicing
        start_x = max(0, c_x - p_x // 2)
        end_x = min(H, start_x + p_x)

        start_y = max(0, c_y - p_y // 2)
        end_y = min(W, start_y + p_y)

        start_z = max(0, c_z - p_z // 2)
        end_z = min(D, start_z + p_z)

        return volume[:, start_x:end_x, start_y:end_y, start_z:end_z]
    
    def __normalize(self, volume):
        """Normalize input to range [0, 1]."""
        min_val = np.min(volume)
        max_val = np.max(volume)
        if max_val > min_val:
            return (volume - min_val) / (max_val - min_val)
        return volume