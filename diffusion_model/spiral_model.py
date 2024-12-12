import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torch.amp import autocast, GradScaler
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from collections import namedtuple
from diffusion_model.dataset import ToyDataset, ToyDataModule

# from sampler import euler_sampler

# Create Diffusion Model, with training and inference functions
# Inference function should take as input, an array of timesteps

data_dimensions = 2


# Specifically the neural net that will be the ODE
class FlowModel(nn.Module):
    def __init__(self, layers, data_dimensions):
        super(FlowModel, self).__init__()
        self.data_dimensions = data_dimensions
        in_size = data_dimensions + 1
        # layers = [(in_size-1) // 8, (in_size-1) // 8]
        # [in_size, 4096, 2048, 1024, 512, 1024, 2048, 4096, in_size - 1]
        layers = [in_size] + layers + [in_size - 1]
        self.seq = nn.Sequential()
        for i in range(len(layers) - 1):
            self.seq.append(nn.Linear(layers[i], layers[i + 1]))
            # if i == len(layers) - 2:
            #    self.seq.append(nn.Sigmoid())
            # else:
            if i < len(layers) - 2:
                self.seq.append(nn.ReLU())

    def forward(self, x, t):
        #print(x.size())
        #batch_size = x.size()[0]
        #reshaped_x = x.reshape(batch_size, self.img_size * self.img_size *3)
        #print(reshaped_x.size())
        #print(t.size())
        print(x.shape, t.shape)
        combined = torch.cat([x, t], dim=1) #torch.tensor([t[0]], device='cuda')
        #print(combined.size())
        output = self.seq(combined)
        return output


# Represents the external parts of the diffusion model


class SpiralDiffusionModel(pl.LightningModule):
    def __init__(
        self,
        model_size="small",
        learning_rate=1e-4,
        num_noise_samples=1,
        loss_type="mse",
        layers=[1024, 1024, 1024],
        total_epochs=10000,
        print_debug=False,
    ):
        super(SpiralDiffusionModel, self).__init__()
        self.layers = layers
        self.model = FlowModel(layers=layers, data_dimensions=data_dimensions)
        #self.nn_model = self.model
        self.learning_rate = learning_rate
        self.num_noise_samples = num_noise_samples
        self.loss_type = loss_type
        self.total_epochs = total_epochs
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "kl":
            self.criterion = nn.KLDivLoss(reduction="batchmean")
        else:
            raise ValueError("Invalid loss_type. Choose 'mse' or 'kl'.")
        self.scaler = GradScaler()
        self.train_loss = MeanMetric()
        # self.noise = torch.randn(batch.size(), device=self.device)

    def get_model(self):
        return self.model

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        batch_size = batch.size(0)
        # batch = batch.repeat_interleave(self.num_noise_samples, dim=0)
        noise = torch.randn(batch.size(), device=self.device)
        t_shape = [batch.size(0)] + [1] * (batch.dim() - 1)
        t = torch.rand(t_shape, device=self.device)
        Xt = t * batch + (1 - t) * noise
        dot_Xt = batch - noise  # derivative of Xt

        with autocast(device_type="cuda"):
            v_t = self.model(Xt, t)
            if self.loss_type == "mse":
                loss = self.criterion(v_t, dot_Xt)
            elif self.loss_type == "kl":
                loss = self.criterion(v_t.log_softmax(dim=1), dot_Xt.softmax(dim=1))

        # Log the loss
        self.train_loss.update(loss)
        self.log("train_loss", self.train_loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )  # , weight_decay=1e-5)
        scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=self.total_epochs, power=2
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# Main training script
if __name__ == "__main__":
    # Configuration
    config = {
        "model_size": "large",
        "batch_size": 10000,
        "learning_rate": 1e-5,
        "num_epochs": 10000,
        "csv_file": "pokemon_static_image_dataset_2.csv",
        "num_gpus": 8,
        "loss_type": "mse",  # Options: 'mse' or 'kl'
        "print_debug": True,  # Toggle for printing debug information
        "layers": [1024, 1024, 1024],
    }

    # Instantiate the model, data module, and trainer
    checkpoint_path = "../model_files5/toy_model-epoch=4999.ckpt"
    model = SpiralDiffusionModel.load_from_checkpoint(checkpoint_path)
    # model = SpiralDiffusionModel(model_size=config['model_size'], layers=config['layers'], learning_rate=config['learning_rate'], loss_type=config['loss_type'], total_epochs=config['num_epochs'], print_debug=config['print_debug'])
    data_module = ToyDataModule(
        csv_file=config["csv_file"],
        batch_size=config["batch_size"],
        dimension=data_dimensions,
        n_samples=80000,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="../model_files6/",  #'/scratch/aadarshnarayan/models/',  # Directory to save the models
        filename="toy_model-{epoch:02d}",  # Filename format
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=500,  # Save every 5 epochs
    )
    # euler_callback = EulerSamplerCallback(save_dir='generated_images/', every_n_epochs=100)

    # Define the trainer
    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        accelerator="gpu",
        devices=config["num_gpus"],
        precision="16-mixed",  # Updated precision
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],  # , euler_callback]
    )

    # Start training
    trainer.fit(model, datamodule=data_module)
    # save_model(model, 500)
