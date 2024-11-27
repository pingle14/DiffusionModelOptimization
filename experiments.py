from diffusion_model.dataset import ToyDataset, generate_saved_data
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from diffusion_model.sampler import euler_sampler
import torch.nn.functional as F
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import csv
from torch.utils.data import random_split
from torchmetrics import MeanMetric
from pytorch_lightning.callbacks import ModelCheckpoint
from diffusion_model.model import DiffusionModel


# Define the Neural Network model
# Input = Noise, Output = Timesteps (k)
class GaussianNoiseNN(nn.Module):
    def __init__(self, input_size=2, output_nTimesteps=100):
        super(GaussianNoiseNN, self).__init__()

        # Define the layers of the network
        self.fc1 = nn.Linear(input_size, 64)  # First hidden layer with 64 units
        self.fc2 = nn.Linear(64, 128)  # Second hidden layer with 128 units
        self.fc3 = nn.Linear(128, output_nTimesteps)  # Output layer with k units

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Sigmoid to constrain output between [0, 1]

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Pass input through the first hidden layer
        x = self.relu(self.fc2(x))  # Pass through the second hidden layer
        # Apply sigmoid to output to ensure it's between [0, 1]
        x = self.sigmoid(self.fc3(x))

        # Post-processing to make sure outputs are distinct
        x = self.make_distinct(x)

        return x

    def make_distinct(self, x):
        # Sort the output to ensure it's ordered (you can also add noise to break ties)
        sorted_x, _ = torch.sort(x, dim=1)

        # Optionally perturb the values slightly to ensure distinctness (avoid exact duplicates)
        epsilon = 1e-6  # Small value to break ties
        sorted_x += epsilon * torch.arange(sorted_x.size(1), device=sorted_x.device)

        # Normalize back to [0, 1] range (if needed)
        sorted_x = (sorted_x - sorted_x.min()) / (sorted_x.max() - sorted_x.min())

        return sorted_x


# Custom loss function
class TimestepLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(TimestepLoss, self).__init__()
        self.alpha = alpha  # Controls the importance of the distinctness penalty
        self.beta = beta  # Controls importance of actual objective function

    def forward(self, output_timesteps, output_generation, target_generation):
        # Loss 1: Penalize outputs that are outside the [0, 1] range (just as a safety check)
        range_loss = torch.sum(
            torch.clamp(output_timesteps, min=0.0, max=1.0) - output_timesteps
        )

        # Loss 2: Penalize if the values are too close to each other (distinctness penalty)
        # Compute pairwise differences between outputs
        pairwise_diff = torch.triu(
            torch.abs(output_timesteps[:, None] - output_timesteps), diagonal=1
        )
        distinctness_loss = torch.sum(
            torch.exp(-pairwise_diff)
        )  # Penalize small differences

        # Loss 3: Actual Objective Function
        #print(output_generation.shape)
        #print(target_generation.shape)
    
        adjusted_mse = F.mse_loss(output_generation, target_generation)

        # Final custom loss: combine all components with respective weights
        loss = self.alpha * distinctness_loss + self.beta * adjusted_mse + range_loss
        return loss

class CSVDataset(Dataset):
    def __init__(self, df):
        df = pd.read_csv("diffusion_model/data.csv")
        df.drop(columns=['Iteration'], inplace=True)
        self.input_noise = np.array(df[['Z1', 'Z2']])
        self.target_generation = np.array(df[['X1','X2']])
        self.n_samples = len(self.input_noise)
        

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Return the data point and its corresponding parameter t value
        return {
            "datapoint": torch.tensor(
                    self.input_noise[idx], dtype=torch.float32
                ),
            "label":torch.tensor(
                    self.target_generation[idx], dtype=torch.float32
                )
        }


# DataModule to handle data loading
class CSVDataModule(pl.LightningDataModule):
    def __init__(
        self, csv_file, batch_size=128, train_val_test_split=[0.8, 0.1, 0.1]
    ):
        super().__init__()
        ### TODO: sample datapoints from the toy function, either pre-computed, or create them here
        # self.csv_file = csv_file
        self.batch_size = batch_size
        self.df = pd.read_csv(csv_file)
        self.train_val_test_split = train_val_test_split

    def setup(self, stage=None):
        # df = pd.read_csv(self.csv_file)
        self.dataset = CSVDataset(
            df=self.df
        )
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, self.train_val_test_split)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )


epochs = 1000
device = "cuda"


class TimseStepSelectorModule(pl.LightningModule):
    def __init__(self, diffusion_model, input_size=2, output_nTimesteps=100, loss_fn=TimestepLoss(alpha=0.0, beta=1.0), learning_rate=1e-4):
        super(TimseStepSelectorModule, self).__init__()
        self.model = GaussianNoiseNN(input_size=input_size, output_nTimesteps=output_nTimesteps)
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.train_loss = MeanMetric()
        self.validation_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.scaler = GradScaler()
        self.diffusion_model = diffusion_model
        for param in self.diffusion_model.parameters():
            param.requires_grad = False

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        with autocast(device_type='cuda'):
            input_noise = batch['datapoint']
            target_generations = batch['label']
            output_timesteps = self.model(input_noise)
            output_generations = euler_sampler(
                self.diffusion_model, xt=input_noise, time_steps=output_timesteps, device=device
            )
            loss = self.loss_fn(output_timesteps, output_generations, target_generations)
       
        # Log the loss
        self.train_loss.update(loss)
        self.log('train_loss', self.train_loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        with autocast(device_type='cuda'):
            input_noise = batch['datapoint']
            target_generations = batch['label']
            output_timesteps = self.model(input_noise)
            output_generations = euler_sampler(
                self.diffusion_model, xt=input_noise, time_steps=output_timesteps, device=device
            )
            loss = self.loss_fn(output_timesteps, output_generations, target_generations)
       
        # Log the loss
        self.validation_loss.update(loss)
        self.log('validation_loss', self.validation_loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        with autocast(device_type='cuda'):
            input_noise = batch['datapoint']
            target_generations = batch['label']
            output_timesteps = self.model(input_noise)
            output_generations = euler_sampler(
                self.diffusion_model, xt=input_noise, time_steps=output_timesteps, device=device
            )
            loss = self.loss_fn(output_timesteps, output_generations, target_generations)
       
        # Log the loss
        self.test_loss.update(loss)
        self.log('test_loss', self.test_loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=10000, power=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }


# Main training script
if __name__ == '__main__':
    # Configuration
    config = {
        'batch_size': 10000,
        'learning_rate': 1e-4,
        'num_epochs': 5000,
        'csv_file': 'diffusion_model/data.csv',
        'num_gpus': 8
    }

    # Instantiate the model, data module, and trainer
    #checkpoint_path =  
    diffy_checkpoint_path = "model_files4/toy_model-epoch=1999.ckpt"
    diffusion_model = DiffusionModel.load_from_checkpoint(diffy_checkpoint_path)
    #DiffusionModel(model_size=config['model_size'], layers=config['layers'], learning_rate=config['learning_rate'], loss_type=config['loss_type'], print_debug=config['print_debug'])
    model = TimseStepSelectorModule(diffusion_model=diffusion_model)
    data_module = CSVDataModule(csv_file=config['csv_file'], batch_size=config['batch_size'])

    checkpoint_callback = ModelCheckpoint(
        dirpath='time_model_files/', #'/scratch/aadarshnarayan/models/',  # Directory to save the models
        filename='time-model-{epoch:02d}',  # Filename format
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=500  # Save every 5 epochs
    )
    #euler_callback = EulerSamplerCallback(save_dir='generated_images/', every_n_epochs=100)

    # Define the trainer
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        accelerator='gpu',
        devices=config['num_gpus'],
        precision='16-mixed',  # Updated precision
        log_every_n_steps=1,
        callbacks=[checkpoint_callback]#, euler_callback]
    )

    # Start training
    trainer.fit(model, datamodule=data_module)
    #save_model(model, 500)