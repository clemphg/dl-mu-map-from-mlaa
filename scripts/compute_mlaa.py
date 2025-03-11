"""
Generate MLAA reconstructions (tof)
"""

import os
from tqdm import tqdm
import numpy as np
import pickle as pkl

import torch

from src.algo.mlaa import MLAA
from src.utils.projector import Projector
from src.utils.data.image_dataset import ImageDataset

def save_image(lambda_mlaa, mu_mlaa, save_path, id_patient):
    full_save_path = os.path.join(save_path, f"{id_patient}.npy")
    stacked = torch.stack([lambda_mlaa, mu_mlaa]).cpu().numpy()
    np.save(full_save_path, stacked)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path_mlaa = ""
    path_data = ""

    dataset = ImageDataset(path_data=path_data,
                           nb_slices_volume=128)
    
    # initializations
    projector = Projector(use_tof=False, use_res_model=True)
    tof_projector = Projector(use_tof=True, use_res_model=True)
    
    mlaa = MLAA(projector=projector,
                tof_projector=tof_projector,
                is_tof=True)
    n_iter = 100

    # initialize mu map
    with open("/home/cphung/Data/CHU_POITIERS/mean_attenuation_128x256x256.pkl", 'rb') as f:
        mean_mu = pkl.load(f)['mean']
    mu_init = torch.full((128, 256, 256), mean_mu)

    # TODO: loop and compute MLAA

    for id_patient in range(len(dataset)):
        image = dataset[id_patient]

        pet_img = image[0].to(device)
        mu_img = image[1].to(device)

        # compute projection
        tau = 0.001
        bckg_pet = 0

        # compute attenuation sinogram
        x_att_fwd = projector.project(mu_img)
        att_sino = torch.exp(-x_att_fwd)

        proj_act = tof_projector.transform(pet_img)
        att_sino = att_sino.unsqueeze(-1).repeat(1, 1, 1, proj_act.shape[-1])

        # forward project with resolution and attenuation model
        y = torch.clamp(tau * (att_sino * proj_act + bckg_pet), min=0)
        ybar = torch.poisson(y)

        # solve mlaa
        lambda_mlaa, mu_mlaa = mlaa.solve(y=ybar,
                                          mu_init=mu_init,
                                          tau=tau,
                                          bckg_pet=bckg_pet,
                                          n_iter=n_iter,
                                          display=False)
        
        # save reconstruction
        save_image(lambda_mlaa, mu_mlaa, save_path_mlaa, id_patient)


if __name__=="__main__":
    main()