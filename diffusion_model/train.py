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

# Dataset definition
class PokemonDataset(Dataset):
    def __init__(self, df, image_size=(96, 96), transform=None):
        self.image_size = image_size
        self.transform = transform
        self.images = df.iloc[:, :-1].values.astype(np.uint8).reshape(-1, *image_size, 3)
        self.labels = df.iloc[:, -1].values
        if self.transform:
            self.images = [(2 * (image.transpose(2, 0, 1) / 255.0) - 1) for image in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        return torch.tensor(image, dtype=torch.float32)

# UNet Model
class UNet(nn.Module):
    def __init__(self, print_debug=False):
        super(UNet, self).__init__()
        self.print_debug = print_debug
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.dec3_upconv = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3_conv1 = nn.Conv2d(512, 256, kernel_size=3, padding='valid')
        self.dec3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding='valid')
        
        self.dec2_upconv = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding='valid')
        self.dec2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding='valid')
        
        self.dec1_upconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding='valid')
        self.dec1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding='valid')
        # Output layer
        self.out = nn.Conv2d(64, 3, kernel_size=1)
        self.padding = nn.ConstantPad2d((14, 14, 14, 14), 0)  # Padding to reach 96x96

    def forward(self, x, t):
        if self.print_debug:
            print(f'Input size: {x.size()}')
        t_expanded = t.expand(-1, 1, x.size(2), x.size(3))
        xt = torch.cat((x, t_expanded), dim=1)
        # Encoder path
        enc1 = self.enc1(xt)
        if self.print_debug:
            print(f'After enc1: {enc1.size()}')
        enc2 = self.enc2(enc1)
        if self.print_debug:
            print(f'After enc2: {enc2.size()}')
        enc3 = self.enc3(enc2)
        if self.print_debug:
            print(f'After enc3: {enc3.size()}')
        # Bottleneck
        bottleneck = self.bottleneck(enc3)
        if self.print_debug:
            print(f'After bottleneck: {bottleneck.size()}')
        # Decoder path
        # Decoder block 3
        dec3 = self.dec3_upconv(bottleneck)
        enc3_resized = nn.functional.interpolate(enc3, size=(dec3.shape[2], dec3.shape[3]), mode='nearest')
        dec3 = torch.cat((dec3, enc3_resized), dim=1)
        dec3 = nn.functional.relu(self.dec3_conv1(dec3))
        dec3 = nn.functional.relu(self.dec3_conv2(dec3))
        if self.print_debug:
            print(f'After dec3: {dec3.size()}')
        # Decoder block 2
        dec2 = self.dec2_upconv(dec3)
        enc2_resized = nn.functional.interpolate(enc2, size=(dec2.shape[2], dec2.shape[3]), mode='nearest')
        dec2 = torch.cat((dec2, enc2_resized), dim=1)
        dec2 = nn.functional.relu(self.dec2_conv1(dec2))
        dec2 = nn.functional.relu(self.dec2_conv2(dec2))
        if self.print_debug:
            print(f'After dec2: {dec2.size()}')
        # Decoder block 1
        dec1 = self.dec1_upconv(dec2)
        enc1_resized = nn.functional.interpolate(enc1, size=(dec1.shape[2], dec1.shape[3]), mode='nearest')
        dec1 = torch.cat((dec1, enc1_resized), dim=1)
        dec1 = nn.functional.relu(self.dec1_conv1(dec1))
        dec1 = nn.functional.relu(self.dec1_conv2(dec1))
        if self.print_debug:
            print(f'After dec1: {dec1.size()}')
        # Output
        output = self.out(dec1)
        output = self.padding(output)
        if self.print_debug:
            print(f'Output size: {output.size()}')
        return output

# LightningModule definition
class DiffusionModel(pl.LightningModule):
    def __init__(self, model_size='small', learning_rate=1e-2, num_noise_samples=1, loss_type='mse', print_debug=False):
        super(DiffusionModel, self).__init__()
        self.model = UNet(print_debug=print_debug)
        self.learning_rate = learning_rate
        self.num_noise_samples = num_noise_samples
        self.loss_type = loss_type
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'kl':
            self.criterion = nn.KLDivLoss(reduction='batchmean')
        else:
            raise ValueError("Invalid loss_type. Choose 'mse' or 'kl'.")
        self.scaler = GradScaler()
        self.train_loss = MeanMetric()

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        images = batch
        batch_size = images.size(0)
        images = images.repeat_interleave(self.num_noise_samples, dim=0)
        noise = torch.randn_like(images, device=self.device)
        t = torch.rand(images.size(0), 1, 1, 1, device=self.device)
        Xt = t * images + (1 - t) * noise
        dot_Xt = images - noise  # derivative of Xt
        
        with autocast(device_type='cuda'):
            v_t = self.model(Xt, t)
            if self.loss_type == 'mse':
                loss = self.criterion(v_t, dot_Xt)
            elif self.loss_type == 'kl':
                loss = self.criterion(v_t.log_softmax(dim=1), dot_Xt.softmax(dim=1))
        
        # Log the loss
        self.train_loss.update(loss)
        self.log('train_loss', self.train_loss, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        return optimizer

# DataModule to handle data loading
class PokemonDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, batch_size=128, image_size=(96, 96)):
        super().__init__()
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (255.0,)),  # Normalize to [0, 1]
            transforms.Normalize((0.5,), (0.5,))    # Normalize to [-1, 1]
        ])

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_file)
        self.dataset = PokemonDataset(df, image_size=self.image_size, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
import pytorch_lightning as pl
import torch
import numpy as np
import matplotlib.pyplot as plt

# Custom Callback for Euler Sampling and Image Generation
class EulerSamplerCallback(pl.Callback):
    def __init__(self, save_dir='generated_images/', every_n_epochs=5, num_samples=4, num_steps=100):
        super().__init__()
        self.save_dir = save_dir  # Directory to save generated images
        self.every_n_epochs = every_n_epochs  # How often to generate images
        self.num_samples = num_samples  # Number of samples to generate
        self.num_steps = num_steps  # Number of Euler steps

    def on_epoch_end(self, trainer, pl_module):
        # Run the Euler sampling process every n epochs
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Generate random noise as input
            device = pl_module.device
            x0 = torch.randn(self.num_samples, 3, 96, 96, device=device)  # Shape: [num_samples, channels, height, width]
            x0 = torch.clamp(x0, -1, 1)  # Clamp noise to range [-1, 1]

            # Run the Euler sampler
            results = self.euler_sampler(pl_module, x0, num_steps=self.num_steps)

            # Get the final generated images
            generated_images = results.xt  # Shape: [num_samples, channels, height, width]
            
            # Save the generated images
            self.save_generated_images(generated_images, trainer.current_epoch)

    def euler_sampler(self, model, x0, num_steps=100):
        xt = x0.clone()
        xtraj = [xt]
        x1_pred_traj = []

        with torch.no_grad():
            for step in range(num_steps):
                t = step / num_steps
                step_size = 1 / num_steps
                t_ones = t * torch.ones(xt.shape[0], 1, 1, 1).to(x0.device)

                v_t = model(xt, t_ones)

                x1_pred = xt + (1 - t) * v_t
                xt = xt + step_size * v_t

                x1_pred_traj.append(x1_pred)
                xtraj.append(xt)

        Results = namedtuple('Results', ['xt', 'xtraj', 'x1_pred_traj'])
        return Results(xt, xtraj, x1_pred_traj)

    def save_generated_images(self, images, epoch):
        # Save each image as PNG
        for i in range(images.size(0)):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
            img = np.clip(img * 0.5 + 0.5, 0, 1)  # Convert from [-1,1] to [0,1]

            # Save the image
            plt.imsave(f'{self.save_dir}generated_epoch_{epoch}_image_{i+1}.png', img)



# Main training script
if __name__ == '__main__':
    # Configuration
    config = {
        'model_size': 'large',
        'batch_size': 32,
        'learning_rate': 1e-2,
        'num_epochs': 500,
        'csv_file': 'pokemon_static_image_dataset_2.csv',
        'num_gpus': 8,
        'loss_type': 'mse',  # Options: 'mse' or 'kl'
        'print_debug': True,  # Toggle for printing debug information
    }

    # Instantiate the model, data module, and trainer
    model = DiffusionModel(model_size=config['model_size'], learning_rate=config['learning_rate'], loss_type=config['loss_type'], print_debug=config['print_debug'])
    data_module = PokemonDataModule(csv_file=config['csv_file'], batch_size=config['batch_size'])

    checkpoint_callback = ModelCheckpoint(
        dirpath='/scratch/aadarshnarayan/models/',  # Directory to save the models
        filename='pokemon_model-{epoch:02d}',  # Filename format
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=500  # Save every 5 epochs
    )
    euler_callback = EulerSamplerCallback(save_dir='generated_images/', every_n_epochs=100)

    # Define the trainer
    trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        accelerator='gpu',
        devices=config['num_gpus'],
        precision='16-mixed',  # Updated precision
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, euler_callback]
    )

    # Start training
    trainer.fit(model, datamodule=data_module)
    #save_model(model, 500)