# TODO: create an abstracte trainer
#       in it, store all the stuff that can be shared
#       with many type of trainers, such as optimizer,
#       gradient clipper, scheduler, dataloader etc.

import os
from tqdm import tqdm
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils.train_utils import get_logger, update_ema

class Trainer():

    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        model: nn.Module,
        loss: nn.Module,
        optimizer: nn.Module,
        scheduler: nn.Module = None,
        epochs: int = 100,
        start_epoch: int = 0,
        grad_clip: float = 1e-1,
        weights_dir: str = './weights',
        log_dir: str = './logs',
        ema_rate: float = None,
        checkpoint_freq: int = 10,
        device: str = 'cuda'
    ):
        super().__init__()

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.model = model
        self.loss = loss
        
        self.epochs = epochs
        self.start_epoch = start_epoch

        self.grad_clip = grad_clip
        if self.grad_clip is not None:
            self.grad_clipper = lambda model_weights: torch.nn.utils.clip_grad_norm_(model_weights, grad_clip)

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.weights_dir = weights_dir
        self.log_dir = log_dir

        self.checkpoint_freq = checkpoint_freq

        if not os.path.exists(weights_dir):
            print(f"Created weights directory {weights_dir}")
            os.makedirs(weights_dir)

        if not os.path.exists(log_dir):
            print(f"Created logs directory {log_dir}")
            os.makedirs(log_dir)
        
        # logging
        self.logger = get_logger(log_dir=log_dir)
        self.tensorboard_writer = SummaryWriter(log_dir=log_dir)

        # ema
        self.ema_rate = ema_rate
        if self.ema_rate is not None:
            self.ema_params = [p.clone().detach() for p in self.model.parameters()]

        self.device = device

    def update_ema(self):
        """
        Update EMA parameters to be closer to the current model parameters.
        """
        update_ema(
            target_params=self.ema_params, 
            source_params=self.model.parameters(), 
            rate=self.ema_rate
        )

    def save_model_weights(self, epoch: int) -> None:
        """
        Save model weights to a file.
        """
        if self.ema_rate is not None:
            weights = {name: p.clone() for name, p in zip(self.model.state_dict().keys(), self.ema_params)}
            model_path = os.path.join(self.weights_dir, f"ema_model_epoch_{epoch}.pth")
        else:
            weights = self.model.state_dict()
            model_path = os.path.join(self.weights_dir, f"model_epoch_{epoch}.pth")

        torch.save(weights, model_path)
        print(f"{'EMA m' if self.ema_rate is not None else 'M'}odel weights saved to {model_path}")

    def one_step(self, packed):
        
        inputs, targets = packed
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        outputs = self.model(inputs)

        return outputs, targets


    def train(self) -> None:

        best_val_loss = 1e6
        best_val_loss_epoch = -1

        for epoch in range(self.start_epoch, self.epochs):

            # Training
            self.model.train()
            cum_loss = 0.0

            pbar = tqdm(self.train_dataloader, desc=f"Training epoch {epoch+1}/{self.epochs}", ncols=100)
            for batch_id, packed in enumerate(pbar):
                
                x, y = self.one_step(packed)

                loss = self.loss(x, y)
                cum_loss += loss.item()

                pbar.set_postfix(loss_train=cum_loss/(batch_id+1))

                # optimization steps
                loss.backward()
                if self.grad_clip is not None:
                    self.grad_clipper(self.model.parameters())
                self.optimizer.step()
                self.optimizer.zero_grad()

                # EMA update
                if self.ema_rate is not None:
                    self.update_ema()

            avg_loss_train = cum_loss/len(self.train_dataloader)


            if (epoch+1) % self.checkpoint_freq == 0:
                self.save_model_weights(epoch+1)

            # # scheduler
            # if self.scheduler is not None:
            #     self.scheduler.step()

            # # log
            # self.logger.info(f"Epoch {epoch+1}/{self.epochs} - Train loss: {avg_loss_train:.5f}")

            # # tensorboard log
            # if self.tensorboard_writer:
            #     self.tensorboard_writer.add_scalar('Loss/train', avg_loss_train, epoch+1)

            # Validation
            self.model.eval()
            cum_loss_val = 0.0

            val_pbar = tqdm(self.val_dataloader, desc=f"Validation epoch {epoch+1}/{self.epochs}", ncols=100)
            with torch.no_grad():
                for batch_id, packed in enumerate(val_pbar):
                    x, y = self.one_step(packed)

                    loss_val = self.loss(x, y)
                    cum_loss_val += loss_val.item()
        
                    val_pbar.set_postfix(loss_val=cum_loss_val/(batch_id+1))
                
            # Saving weights
            avg_loss_val = cum_loss_val/len(self.val_dataloader)

            if avg_loss_val <= best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.5f} (epoch {best_val_loss_epoch+1}) to {avg_loss_val:.5f}.")
                #self.save_model_weights(epoch+1)
                best_val_loss = avg_loss_val
                best_val_loss_epoch = epoch
            elif best_val_loss==1e6 and epoch+1==self.epochs:
                print(f"No improvement of the loss over validation during training.")
                #self.save_model_weights(epoch+1)
            else:
                print(f"Validation loss did not improve from {best_val_loss:.5f} (epoch {best_val_loss_epoch+1}).")

            # scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # log
            self.logger.info(f"Epoch {epoch+1}/{self.epochs} - Train loss: {avg_loss_train:.5f}, Validation loss: {avg_loss_val:.5f}")

            # tensorboard log
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('Loss/train', avg_loss_train, epoch+1)
                self.tensorboard_writer.add_scalar('Loss/validation', avg_loss_val, epoch+1)

            print()

