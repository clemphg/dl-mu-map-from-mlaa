
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from src.model.unet import UNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parameters
    batch_size = 128
    num_epochs = 10

    lr_initial = 0.001
    lr_decay_rate = 1/2
    lr_decay_freq = 2

    # data
    train_dataloader = None

    # initialization
    model = UNet.to(device)
    loss_fun = nn.L1Loss()
    optimizer = optim.Adam(model.parameters, lr=lr_initial)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_freq, gamma=lr_decay_rate)

    print("\n"+"-"*23+" TRAINING "+"-"*23+"\n")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_dataloader, desc=f"Training epoch {epoch+1}/{num_epochs}", ncols=100)
        for batch_id, (inputs, targets) in enumerate(pbar):

            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = loss_fun(outputs, targets)
            running_loss += loss.item()

            pbar.set_postfix(loss_train=running_loss/(batch_id+1))

            # optimization steps
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()  # adjust learning rate
        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed - Avg Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

    print("Training complete!")

if __name__=="__main__":
    main()