
import os
import sys
from tqdm import tqdm
import pickle as pkl
import dotenv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
dotenv.load_dotenv()

from src.models.unet import UNet
from src.utils.data.patch_dataset import PatchDataset
from src.utils.trainer import Trainer
from src.utils.train_utils import get_logger, save_model_weights

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parameters
    batch_size = 4
    num_epochs = 3830 # adapted to have about 10*9955 total iterations 

    lr_initial = 0.001
    lr_decay_rate = 1/2
    lr_decay_freq = 100

    # data
    path_inputs = os.environ['PATH_MLAA']
    path_targets = os.environ['PATH_REFERENCE']

    path_id_patients = os.environ['PATH_ID_PATIENTS']
    with open(path_id_patients, 'rb') as f:
        id_patients = pkl.load(f)
    id_patients_train = id_patients['train']
    id_patients_valid = id_patients['train_valid']

    print(len(id_patients_train), len(id_patients_valid))

    train_dataset = PatchDataset(id_patients=id_patients_train,
                                 path_inputs=path_inputs,
                                 path_targets=path_targets,
                                 patch_shape=(64,256,256),
                                 nb_slices_volume=64)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    print(len(train_dataset),
          len(train_dataloader))

    valid_dataset = PatchDataset(id_patients=id_patients_valid,
                                 path_inputs=path_inputs,
                                 path_targets=path_targets,
                                 patch_shape=(64,256,256),
                                 nb_slices_volume=64)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)

    print(f"Nb training patients: {len(id_patients_train)}")
    print(f"Nb training volumes: {len(train_dataset)}")
    print(f"Shape of inputs: {train_dataset[0][0].shape}")
    print(f"Shape of targets: {train_dataset[0][1].shape}")

    # initialization
    model = UNet(in_channels=2, out_channels=2).to(device)
    loss_fun = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr_initial)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_freq, gamma=lr_decay_rate)

    trainer = Trainer(train_dataloader=train_dataloader,
                      val_dataloader=valid_dataloader,
                      model=model,
                      loss=loss_fun,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      epochs=num_epochs,
                      start_epoch=0,
                      grad_clip=None,
                      weights_dir="./weights",
                      log_dir="./logs",
                      ema_rate=None,
                      checkpoint_freq=1,
                      device=device)

    print("\n"+"-"*23+" TRAINING "+"-"*23+"\n")
    trainer.train()

if __name__=="__main__":
    main()