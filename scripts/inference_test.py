

import os
import sys
import torch
import dotenv
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
dotenv.load_dotenv()

from src.models.unet import UNet
from src.utils.data.measurements_image_dataset import MeasuresImageDataset
from src.algos.mlem import MLEM
from src.utils.projector import Projector


def linear_weight(depth, type='middle'):
    """Generates a weight that increases towards the center."""
    w = np.linspace(0, 1, depth//2)
    w = np.concatenate([w, w[::-1]])  # Symmetric weight profile
    if type=='start':
        w[:depth//2] = 1
    elif type=='end':
        w[depth//2:] = 1
    return w[:, None, None]  # Reshape for broadcasting

def save_file(filename, save_path, stacked_recon):
    save_filename = os.path.join(save_path, filename)
    if save_filename not in os.listdir(save_path):
        np.save(save_filename, stacked_recon)

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tau = 0.000001
    bckg_pet = 0
    use_tof = True

    n_iter_mlem = 10
    n_iter_mlaa = 10

    kernel_size = 3
    best_epoch_valid = 59
    model_name = "2025-04-02_tau1e-6_k3/"
    full_path_model = os.path.join(os.environ['PATH_MODELS'], model_name)
    path_weights = os.path.join(full_path_model, f"weights/model_epoch_{best_epoch_valid}.pth")

    # PATHS
    path_mlaa_test = os.path.join(os.environ['PATH_SAVE_RECONS'], f"mlaa_test_tau{tau}_{'tof' if use_tof else 'notof'}_{n_iter_mlaa}iter")
    path_measurements_test = os.path.join(os.environ['PATH_MEASUREMENTS'], f"tau{tau}_{'tof' if use_tof else 'notof'}")
    save_path = os.path.join(os.environ[f'PATH_SAVE_RECONS'], f"unet_test_tau{tau}_{'tof' if use_tof else 'notof'}")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # PATIENTS IDS
    path_id_patients = os.environ['PATH_ID_PATIENTS']
    with open(path_id_patients, 'rb') as f:
        id_patients = pkl.load(f)
    id_patients_test = id_patients['test']

    # DATALOADING
    clip_act = 100000
    clip_atn = 0.025

    def norm_pet(pet, clip=False):
        if clip:
            pet = np.clip(pet, a_min=0, a_max=clip_act) 
        return (pet / clip_act) ** (1/3) * 2 # to match norm mu range

    def norm_mu(mu, clip=False):
        if clip:
            mu = np.clip(mu, a_min=0, a_max=clip_atn)
        return mu / 0.015 # / 0.015 mm-1 (skull bone ac at 511 keV)

    def denorm_pet(pet_norm):
        return clip_act * (pet_norm / 2) ** 3 # to match norm mu range

    def denorm_mu(mu, clip=False):
        if clip:
            mu = np.clip(mu, a_min=0, a_max=1)
        return mu * 0.015 # / 0.015 mm-1 (skull bone ac at 511 keV)

    test_dataset = MeasuresImageDataset(id_patients=id_patients_test,
                                        path_inputs=path_mlaa_test,
                                        path_measures=path_measurements_test,
                                        norm_pet=norm_pet,
                                        norm_mu=norm_mu,
                                        return_fn=True)

    # MODEL
    model = UNet(in_channels=2, out_channels=1, kernel_size=kernel_size).to(device)
    model.load_state_dict(torch.load(path_weights, weights_only=True))
    model.eval()

    # PROJECTORS
    projector = Projector(use_tof=False, use_res_model=True)
    tof_projector = Projector(use_tof=True, use_res_model=True)

    mlem = MLEM(projector, tof_projector)


    for packed in test_dataset:

        filename, mlaa_img, ybar = packed
        mlaa_img = mlaa_img.to(device)
        ybar = ybar.to(device)

        # COMPUTE ATTENUATION MAP
        mu_pred = torch.zeros((64, 256, 256))
        weights = torch.zeros((64, 256, 256))

        for i in [0, 16, 32]:
            mu_patch_pred = model(mlaa_img[:, i:i+32].unsqueeze(0)).detach().cpu().squeeze()
            if i==0:
                type='start'
            elif i==32:
                type='end'
            else:
                type = 'middle'
            mu_pred[i:i+32] += mu_patch_pred * linear_weight(32, type=type)
            weights[i:i+32] += linear_weight(32, type=type)

        mu_pred /= torch.where(weights == 0, 1, weights)
        mu_pred = denorm_mu(mu_pred)

        # RECON ACTIVITY
        mu_pred = mu_pred.to(device)
        recon_lambda = mlem.solve(ybar,
                                mu_pred,
                                tau=tau,
                                bckg_pet=bckg_pet,
                                n_iter=n_iter_mlem,
                                use_tof=use_tof,
                                display=False)
        recon_pet = recon_lambda.cpu()

        # SAVE RECON AND PREDICTED MU MAP
        stacked_recon = torch.stack([recon_pet, mu_pred.cpu()], dim=0)
        save_file(filename, save_path, stacked_recon.numpy())


if __name__=="__main__":
    main()