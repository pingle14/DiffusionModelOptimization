import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
import torch.optim as optim
from torchmetrics import MeanMetric
from torch.amp import autocast, GradScaler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
#rom pretrained import PretrainedConvModel
#from lightning_3 import DiffusionModel
from model import DiffusionModel

# # Load the model
# checkpoint_path = '/scratch/aadarshnarayan/models/pokemon_model-epoch=499.ckpt'
# #'models/pokemon_model-epoch=9999.ckpt'  # Update with your actual checkpoint path
# model = DiffusionModel.load_from_checkpoint(checkpoint_path)
# model.eval()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

def generative_denoising_timestep_order(timesteps):
    return reversed(timesteps)

def noising_timestep_order(timesteps):
    return timesteps

# Euler sampling function
def euler_sampler(model, num_samples=4, time_steps=[], device=device):
    xt = torch.randn(num_samples, 3, 96, 96, device=device)
    xtraj = [xt.clone()]

    time_steps = generative_denoising_timestep_order(time_steps)

    with torch.no_grad():
        for i, step in enumerate(time_steps):
            # Time variable t decreases from 1 to 0
            if step == 1:
                break
            # TODO: note that will need to fiddle with dimensions of this tensor
            t = torch.full((xt.shape[0], 1, 1, 1), fill_value=step, device=device)
            denominator = (time_steps[i+1] if i < len(time_steps) - 1 else 1) - time_steps[i]
            step_size = 1.0 / denominator

            v_t = model(xt, t)

            # Update xt using Euler method
            xt = xt + step_size * v_t

            xtraj.append(xt.clone())

    return xt

# # Generate images
# num_samples = 4  # Adjust as needed
# num_steps = 100
# generated_images = euler_sampler(model, num_samples=num_samples, num_steps=num_steps)

# # Save generated images
# save_dir = 'test_images/'
# os.makedirs(save_dir, exist_ok=True)

# for i in range(generated_images.size(0)):
#     img = generated_images[i].detach().cpu()
#     img = (img + 1) / 2  # Scale from [-1,1] to [0,1]
#     img = torch.clamp(img, 0, 1)
#     save_image(img, os.path.join(save_dir, f'generated_image_{i+1}.png'))
