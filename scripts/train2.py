
import os
import sys
from tqdm import tqdm
import pickle as pkl
import dotenv
import glob
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
dotenv.load_dotenv()

from src.models.unet import UNet
from src.models.losses import image_gradient_difference_loss, line_integral_loss
from src.utils.data.patch_dataset import PatchDataset
from src.utils.trainer import Trainer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tau = 0.00001
    kernel_size = 3

    model_name = f"{datetime.datetime.today().strftime('%Y-%m-%d')}_k{kernel_size}_tau{tau}"
    model_path = os.path.join(os.environ['PATH_MODELS'], model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.realpath(__file__), os.path.join(model_path, __file__.split(os.sep)[-1]))

    # parameters
    batch_size = 16
    num_epochs = 500

    lr_initial = 0.001
    lr_decay_rate = 0.975
    lr_decay_freq = 3

    # loss
    beta1 = 1.0
    beta2 = 0.02

    loss_im = nn.L1Loss()
    loss_grad = image_gradient_difference_loss
    loss_lip = line_integral_loss
    
    def full_loss(x, y, beta_1, beta_2):
        l1 = loss_im(x, y)
        l2 = loss_grad(x, y)
        l3 = loss_lip(x, y)
        #print(l1.detach().cpu(), l2.detach().cpu(), l3.detach().cpu())
        return l1 + beta_1 * l2 + beta_2 * l3
    
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

    # data
    path_inputs_train = os.environ[f'PATH_MLAA_TRAIN_TAU{tau}']
    path_inputs_valid = os.environ[f'PATH_MLAA_VALID_TAU{tau}']
    path_targets = os.environ['PATH_REFERENCE']

    path_id_patients = os.environ['PATH_ID_PATIENTS']
    with open(path_id_patients, 'rb') as f:
        id_patients = pkl.load(f)
    id_patients_train = id_patients['train']
    id_patients_valid = id_patients['valid']

    print(len(id_patients_train), len(id_patients_valid))

    patch_shape = (32,64,64)

    train_dataset = PatchDataset(id_patients=id_patients_train,
                                 path_inputs=path_inputs_train,
                                 path_targets=path_targets,
                                 patch_shape=patch_shape,
                                 nb_slices_volume=64,
                                 norm_pet=norm_pet,
                                 norm_mu=norm_mu)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=2)

    print(len(train_dataset),
          len(train_dataloader))

    valid_dataset = PatchDataset(id_patients=id_patients_valid,
                                 path_inputs=path_inputs_valid,
                                 path_targets=path_targets,
                                 patch_shape=patch_shape,
                                 nb_slices_volume=64,
                                 norm_pet=norm_pet,
                                 norm_mu=norm_mu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True, prefetch_factor=2)

    print(f"Nb training patients: {len(id_patients_train)}")
    print(f"Nb training volumes: {len(train_dataset)}")
    print(f"Shape of inputs: {train_dataset[0][0].shape}")
    print(f"Shape of targets: {train_dataset[0][1].shape}")


    # initialization
    model = UNet(in_channels=2, out_channels=1, kernel_size=3).to(device)

    start_epoch = 0
    if os.path.exists(model_path):
        list_of_files = glob.glob(os.path.join(model_path, 'weights', '*')) # * means all if need specific format then *.csv
        if len(list_of_files)>0:
            last_weights_path = max(list_of_files, key=os.path.getctime)
            start_epoch = int(last_weights_path.split('.')[0].split('_')[-1])
            print(f"Resuming from epoch {start_epoch}...")
            model.load_state_dict(torch.load(last_weights_path, weights_only=True))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Nb of params: {pytorch_total_params}\n")

    loss = lambda x, y: full_loss(x, y, beta1, beta2)
    optimizer = optim.Adam(model.parameters(), lr=lr_initial)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_freq, gamma=lr_decay_rate)

    trainer = Trainer(train_dataloader=train_dataloader,
                      val_dataloader=valid_dataloader,
                      model=model,
                      loss=loss,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      epochs=num_epochs,
                      start_epoch=start_epoch,
                      grad_clip=None,
                      weights_dir=os.path.join(model_path, "weights"),
                      log_dir=os.path.join(model_path, "logs"),
                      ema_rate=None,
                      checkpoint_freq=1,
                      device=device)

    print("\n"+"-"*23+" TRAINING "+"-"*23+"\n")
    trainer.train()

if __name__=="__main__":
    main()
