
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataloader

from torch.utils.tensorboard import SummaryWriter

from src.model.unet import UNet
from src.utils.data.patch_dataset import PatchDataset
from src.utils.train_utils import get_logger, save_model_weights

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # log
    weights_dir = './weights',
    log_dir = './logs',
    logger = get_logger(log_dir=log_dir)
    tensorboard_writer = SummaryWriter(log_dir=log_dir)

    # parameters
    batch_size = 128
    num_epochs = 50000 # adapted to have about 10*9955 total iterations 

    lr_initial = 0.001
    lr_decay_rate = 1/2
    lr_decay_freq = 2

    # data
    path_inputs = ""
    path_targets = ""

    id_patients_train = []
    id_patients_valid = []

    train_dataset = PatchDataset(id_patients=id_patients_train,
                                 path_inputs=path_inputs,
                                 path_targets=path_targets,
                                 patch_shape=(32,32,32),
                                 nb_slices_volume=128)
    train_dataloader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    valid_dataset = PatchDataset(id_patients=id_patients_valid,
                                 path_inputs=path_inputs,
                                 path_targets=path_targets,
                                 patch_shape=(32,32,32),
                                 nb_slices_volume=128)
    valid_dataloader = Dataloader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # initialization
    model = UNet.to(device)
    loss_fun = nn.L1Loss()
    optimizer = optim.Adam(model.parameters, lr=lr_initial)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_freq, gamma=lr_decay_rate)

    print("\n"+"-"*23+" TRAINING "+"-"*23+"\n")
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        
        # TRAINING
        pbar = tqdm(train_dataloader, desc=f"Training epoch {epoch+1}/{num_epochs}", ncols=100)
        for batch_id, (inputs, targets) in enumerate(pbar):

            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = loss_fun(outputs, targets)
            running_train_loss += loss.item()

            pbar.set_postfix(loss_train=running_train_loss/(batch_id+1))

            # optimization steps
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()  # adjust learning rate
        avg_loss_train = running_train_loss / len(train_dataloader)

        # VALIDATION
        model.eval()
        running_val_loss = 0.0

        val_pbar = tqdm(valid_dataloader, desc=f"Validation epoch {epoch+1}/{num_epochs}", ncols=100)
        with torch.no_grad():
            for batch_id, packed in enumerate(val_pbar):
                inputs, targets = inputs.to(device), targets.to(device)

                loss = loss_fun(outputs, targets)
                running_val_loss += running_val_loss.item()
    
                val_pbar.set_postfix(loss_val=running_val_loss/(batch_id+1))

        # Saving weights
        avg_loss_val = running_val_loss/len(valid_dataloader)

        if avg_loss_val <= best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.5f} (epoch {best_val_loss_epoch+1}) to {avg_loss_val:.5f}.")
            save_model_weights(model, epoch+1, weights_dir)
            best_val_loss = avg_loss_val
            best_val_loss_epoch = epoch
        elif best_val_loss==1e6 and epoch+1==num_epochs:
            print(f"No improvement of the loss over validation during training.")
            save_model_weights(model, epoch+1, weights_dir)
        else:
            print(f"Validation loss did not improve from {best_val_loss:.5f} (epoch {best_val_loss_epoch+1}).")

        # log
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train loss: {avg_loss_train:.5f}, Validation loss: {avg_loss_val:.5f}")

        # tensorboard log
        if tensorboard_writer:
            tensorboard_writer.add_scalar('Loss/train', avg_loss_train, epoch+1)
            tensorboard_writer.add_scalar('Loss/validation', avg_loss_val, epoch+1)

    print("Training complete!")

if __name__=="__main__":
    main()