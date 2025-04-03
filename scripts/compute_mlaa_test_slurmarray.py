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
from src.utils.data.measurement_dataset import MeasurementDataset

def save_image(lambda_mlaa, mu_mlaa, save_path, filename):
    full_save_path = os.path.join(save_path, filename)
    stacked = torch.stack([lambda_mlaa, mu_mlaa]).cpu().numpy()
    np.save(full_save_path, stacked)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tau = 0.000001
    bckg_pet = 0
    use_tof = False

    n_iter = 10

    clip_act = 100000
    clip_atn = 0.025

    save_path_mlaa = os.path.join(os.environ['PATH_SAVE_RECONS'], f"mlaa_test_tau{tau}_{'tof' if use_tof else 'notof'}_{n_iter}iter")
    path_data = os.path.join(os.environ['PATH_MEASUREMENTS'], f"tau{tau}_{'tof' if use_tof else 'notof'}")

    if not os.path.exists(save_path_mlaa):
        os.makedirs(save_path_mlaa)
        print(f"Created directory {save_path_mlaa}")

    path_id_patients = os.environ['PATH_ID_PATIENTS']
    with open(path_id_patients, 'rb') as f:
        id_patients = pkl.load(f)
    id_patients = id_patients['test']
    print(len(id_patients))

    task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
    id_patients = id_patients[3*task_id:3*(task_id+1)]

    print("id_patients=", id_patients)

    dataset = MeasurementDataset(path_data=path_data,
                                 id_patients=id_patients)
    
    print(len(dataset))
    
    # initializations
    projector = Projector(use_tof=False, use_res_model=True, device=device)
    tof_projector = Projector(use_tof=True, use_res_model=True, device=device)
    
    mlaa = MLAA(projector=projector,
                tof_projector=tof_projector,
                is_tof=use_tof)

    # initialize mu map
    mu_init = torch.full((64, 256, 256), 0.0025) # initialize to mean mu value in training

    #pbar = tqdm(range(len(dataset)), ncols=100)
    for id_patient in range(len(dataset)):

        filename, ybar = dataset[id_patient]
        print(filename)

        if filename in os.listdir(save_path_mlaa):
            continue

        ybar = ybar.to(device)

        # solve mlaa
        lambda_mlaa, mu_mlaa = mlaa.solve(y=ybar,
                                          mu_init=mu_init,
                                          tau=tau,
                                          bckg_pet=bckg_pet,
                                          n_iter=n_iter,
                                          display=False)
        
        # save reconstructions
        save_image(lambda_mlaa, mu_mlaa, save_path_mlaa, filename)


if __name__=="__main__":
    main()
