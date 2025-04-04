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

def save_image(img, save_path, filename):
    full_save_path = os.path.join(save_path, filename)
    stacked = img.cpu().numpy()
    np.save(full_save_path, stacked)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tau = 1e-5
    bckg_pet = 0
    use_tof = False

    clip_act = 100000
    clip_atn = 0.025

    path_data = os.environ['PATH_REFERENCE_TEST']
    save_path_measurements = os.path.join(os.environ['PATH_MEASUREMENTS'], f"tau{tau}_{'tof' if use_tof else 'notof'}")

    if not os.path.exists(save_path_measurements):
        os.makedirs(save_path_measurements)
        print(f"Created directory {save_path_measurements}")

    path_id_patients = os.environ['PATH_ID_PATIENTS']
    with open(path_id_patients, 'rb') as f:
        id_patients = pkl.load(f)
    id_patients = id_patients['test']
    print(len(id_patients))

    dataset = ImageDataset(path_data=path_data,
                           nb_slices_volume=64,
                           id_patients=id_patients)
    
    print(len(dataset))
    
    # initializations
    projector = Projector(use_tof=False, use_res_model=True)
    tof_projector = Projector(use_tof=True, use_res_model=True)

    #pbar = tqdm(range(len(dataset)), ncols=100)
    for id_patient in range(len(dataset)):

        filename, image = dataset[id_patient]

        pet_img = image[0].to(device)
        mu_img = image[1].to(device)

        pet_img = torch.clamp(pet_img, min=0, max=clip_act)
        mu_img = torch.clamp(mu_img, min=0, max=clip_atn)

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

        print(ybar.shape)
        
        # save measurements
        save_image(ybar, save_path_measurements, filename)

if __name__=="__main__":
    main()
