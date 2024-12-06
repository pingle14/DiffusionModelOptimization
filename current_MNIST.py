''' 
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''

from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torch.utils.data.distributed import DistributedSampler
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd
import pytorch_lightning as pl

import torch.distributed as dist
import os



class ResidualConvBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7), # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # x is (noisy) image, c is context label, t is timestep, 
        # context_mask says which samples to block the context on

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        # up2 = self.up1(up1, down2) # if want to avoid add and multiply embeddings
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }


class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_Timesteps, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_Timesteps).items():
            self.register_buffer(k, v)

        self.n_Timesteps = n_Timesteps
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, c):
        """
        this method is used in training, so samples t and noise randomly
        """

        _ts = torch.randint(1, self.n_Timesteps+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_Timesteps)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_Timesteps, context_mask))

    def sample(self, x_i, c_i, context_mask, size, device, guide_w = 0.0, enable_store=True):
        input_noise = []
        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        n_sample = x_i.shape[0]

        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free


        for i in range(self.n_Timesteps, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            # NOTE: when we use our own timesteps, we simply modify this line here, to use our generated timesteps
            t_is = torch.tensor([i / self.n_Timesteps]).to(device)

            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            if (i == self.n_Timesteps):
                input_noise.append(z)
            #input_noise.append(z)

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            #if enable_store and (i%20==0 or i==self.n_Timesteps or i<8):
            #    x_i_store.append(x_i.detach().cpu().numpy())

        #x_i_store = np.array(x_i_store)
        return {'input_noise':input_noise, 'x_i_gen': x_i, 'x_i_gen_store': x_i_store}

    def old_sample(self, n_sample, size, device, guide_w = 0.0, enable_store=True):
        # we follow the guidance sampling scheme described in 'Classifier-Free Diffusion Guidance'
        # to make the fwd passes efficient, we concat two versions of the dataset,
        # one with context_mask=0 and the other context_mask=1
        # we then mix the outputs with the guidance scale, w
        # where w>0 means more guidance

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.arange(0,10).to(device) # context for us just cycles throught the mnist labels
        c_i = c_i.repeat(int(n_sample/c_i.shape[0]))

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(device)

        # double the batch
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1. # makes second half of batch context free

        input_noise = []
        x_i_store = [] # keep track of generated steps in case want to plot something 
        print()
        for i in range(self.n_Timesteps, 0, -1):
            print(f'sampling timestep {i}',end='\r')
            t_is = torch.tensor([i / self.n_Timesteps]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)

            # double batch
            x_i = x_i.repeat(2,1,1,1)
            t_is = t_is.repeat(2,1,1,1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            input_noise.append(z)

            # split predictions and compute weighting
            eps = self.nn_model(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1+guide_w)*eps1 - guide_w*eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if enable_store and (i%20==0 or i==self.n_Timesteps or i<8):
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)

        return {'input_noise':input_noise, 'x_i_gen': x_i, 'x_i_gen_store': x_i_store}

  
class LitSampler(pl.LightningModule):
    def __init__(self, ddpm_model, n_sample, size):
        super(LitSampler, self).__init__()
        self.ddpm_model = ddpm_model
        self.n_sample = n_sample
        self.size = size

    
    def predict_step(self, batch, batch_idx):
        """
        Handles a batch during the predict phase.
        """
        x_i, c_i, context_mask = batch['x_i'], batch['c_i'], batch['context_mask'] # Unpack theddp batch
        predictions = self.ddpm_model.sample(x_i, c_i, context_mask, self.size, self.device, guide_w=0.0, enable_store=True)

        noise = predictions['input_noise'][0]
        gen = predictions['x_i_gen']

        data = torch.cat([noise, gen, c_i], dim=1)

        return data

        """        gathered_predictions = [torch.zeros_like(data) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_predictions, data)
        """
        # Only return the gathered predictions on the main process
        """if torch.distributed.get_rank() == 0:
            print('hello, world! my rank is 0')
            return torch.cat(gathered_predictions, dim=0)
        else:
            return None"""
    
    
class SamplingDataset(Dataset):
    def __init__(self, n_sample, size, n_classes=10, device="cpu"):
        """
        Args:
            n_sample (int): Total number of samples.
            size (tuple): Shape of the data (channels, height, width).
            n_classes (int): Number of classes for context (e.g., MNIST has 10).
            device (str): Device to generate the data on.
        """
        self.n_sample = n_sample
        self.size = size
        self.n_classes = n_classes
        self.device = device

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx):
        """
        Generates a single sample with its context and context_mask.
        """
        x_i = torch.randn(self.size).to(self.device)  # x_T ~ N(0, 1), sample initial noise
        c_i = torch.tensor(idx % self.n_classes).to(self.device)

        # don't drop context at test time
        context_mask = torch.zeros_like(c_i).to(self.device)

        # double the batch
        #c_i = c_i.repeat(2)
        #context_mask = context_mask.repeat(2)
        #context_mask[n_sample:] = 1. # makes second half of batch context free

        return {'x_i': x_i,
                'c_i': c_i, 
                'context_mask': context_mask}
    
class DummyDataset(Dataset):
    def __init__(self, n_sample, size, num_batches=1):
        self.n_sample = n_sample
        self.size = size
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        return self.n_sample, self.size


def test_mnist(save_dir):
    # hardcoding these here
    n_Timesteps = 400 # 500
    device = "cuda:0"
    n_classes = 10
    n_feat = 128 # 128 ok, 256 better (but slower)
    guidance_wieght = 2.0
    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_Timesteps=n_Timesteps, device=device, drop_prob=0.1)
    ddpm.load_state_dict(torch.load("./model_39.pth"))
    ddpm.to(device)

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

    # TODO: Make sure our model isnt cheating and seeing the test dataset during training
    dataset = MNIST("./data", train=True, download=True, transform=tf)
    #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    # for eval, save an image of currently generated samples (top rows)
    # followed by real images (bottom rows)

    

    # Define the number of samples and their size
    n_sample = 10  # Number of samples
    size = (1, 28, 28)  # Example size: (channels, height, width)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lit_sampler = LitSampler(ddpm, n_sample, size)

    # Create dataset and sampler
    sampling_dataset = SamplingDataset(n_sample, size, device=device)
    #sampler = DistributedSampler(sampling_dataset, shuffle=False)  # Ensure proper GPU splits

    # Create DataLoader
    sampling_dataloader = DataLoader(sampling_dataset, batch_size=n_sample)

    # Define the PyTorch Lightning Trainer with multi-GPU support
    trainer = pl.Trainer(accelerator="gpu", devices="auto")#, strategy='ddp_spawn')  # Adjust 'devices' to the number of GPUs

    # Run sampling on multiple GPUs
    #input_noise, x_gen, _ 
    data = trainer.predict(lit_sampler, sampling_dataloader, return_predictions=True)[0].to('cuda')

    #print(type(data))

    #print(f"Current backend: {dist.get_backend()}")
    #print(f"Current device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    #print(f"Rank: {dist.get_rank()}")
    torch.save(data, f'output_{dist.get_rank()}.file')

 
if __name__ == "__main__":
    #if not dist.is_initialized():
    #   dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    test_mnist("./")
    pass

