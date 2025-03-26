

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import lightning as L


class LitUNet(L.LightningModule):
    def __init__(self, model, loss_fun, lr):
        super().__init__()

        self.model = model
        self.loss_fun = loss_fun

        self.lr = lr


    def training_step(self, batch, batch_id):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss_fun(outputs, targets)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        val_loss = self.loss_fun(outputs, targets)
        self.log("val_loss", val_loss)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    
