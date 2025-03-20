"""
Generate MLAA reconstructions (tof)
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

from src.algos.mlaa import MLAA
from src.utils.projector import Projector
from src.utils.data.image_dataset import ImageDataset

def save_image(lambda_mlaa, mu_mlaa, save_path, filename):
    full_save_path = os.path.join(save_path, filename)
    stacked = torch.stack([lambda_mlaa, mu_mlaa]).cpu().numpy()
    np.save(full_save_path, stacked)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path_mlaa = os.environ['PATH_MLAA']
    path_data = os.environ['PATH_REFERENCE']

    if not os.path.exists(save_path_mlaa):
        os.makedirs(save_path_mlaa)
        print(f"Created directory {save_path_mlaa}")

    path_id_patients = os.environ['PATH_ID_PATIENTS']
    with open(path_id_patients, 'rb') as f:
        id_patients = pkl.load(f)
    id_patients = id_patients['train']
    #id_patients = id_patients['train_valid']
    print(len(id_patients))

    dataset = ImageDataset(path_data=path_data,
                           nb_slices_volume=64,
                           id_patients=id_patients)
    
    # initializations
    projector = Projector(use_tof=False, use_res_model=True)
    tof_projector = Projector(use_tof=True, use_res_model=True)

    tau = 0.0001
    bckg_pet = 0
    use_tof = True
    
    mlaa = MLAA(projector=projector,
                tof_projector=tof_projector,
                is_tof=use_tof)
    n_iter = 50

    # initialize mu map
    mu_init = torch.full((64, 256, 256), 0.01) # initialize to water LAC in mm-1


    for id_patient in tqdm(range(len(dataset)), ncols=100):

        filename, image = dataset[id_patient]

        if filename in os.listdir(save_path_mlaa):
            continue

        pet_img = image[0].to(device)
        mu_img = image[1].to(device)

        # compute projection

        # compute attenuation sinogram
        x_att_fwd = projector.transform(mu_img)
        att_sino = torch.exp(-x_att_fwd)

        if use_tof:
            proj_act = tof_projector.transform(pet_img)
            att_sino = att_sino.unsqueeze(-1).repeat(1, 1, 1, proj_act.shape[-1])
        else:
            proj_act = projector.transform(pet_img)

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
        save_image(lambda_mlaa, mu_mlaa, save_path_mlaa, filename)


if __name__=="__main__":
    main()