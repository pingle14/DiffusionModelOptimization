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

# rom pretrained import PretrainedConvModel
# from lightning_3 import DiffusionModel
from diffusion_model.model import DiffusionModel
import matplotlib.pyplot as plt
import pandas as pd

"""# # Load the model
checkpoint_path = "../model_files4/toy_model-epoch=1999.ckpt"  #'/scratch/aadarshnarayan/models/pokemon_model-epoch=499.ckpt'
# #'models/pokemon_model-epoch=9999.ckpt'  # Update with your actual checkpoint path
model = DiffusionModel.load_from_checkpoint(checkpoint_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)"""


def generative_denoising_timestep_order(timesteps):
    return reversed(timesteps)


def noising_timestep_order(timesteps):
    return timesteps


# Euler sampling function
def euler_sampler(model, xt, time_steps=[], device="cuda"):
    xtraj = [xt.clone()]

    # time_steps = generative_denoising_timestep_order(time_steps)

    with torch.no_grad():
        for i, step in enumerate(time_steps):
            # Time variable t decreases from 1 to 0
            # TODO: note that will need to fiddle with dimensions of this tensor
            # TODO: fix step/ t stufffff
            t = step  # TODO: Handle non-tensor case with : torch.full((xt.shape[0], 1), fill_value=step, device=device)
            all_ones = torch.ones_like(time_steps[i], device=device)
            denominator = (
                time_steps[i + 1] if i < len(time_steps) - 1 else all_ones
            ) - time_steps[i]
            # print(denominator)
            step_size = denominator

            v_t = model(xt, t)
            # Update xt using Euler method
            xt = xt + step_size * v_t

            xtraj.append(xt.clone().detach())

    return xt, xtraj


# # # Generate images
# num_samples = 10000  # Adjust as needed
# num_steps = 1000

# xt = torch.randn((num_samples, 2), device=device)
# original_noise = xt.clone().detach()
# generated_datapoints, traj = euler_sampler(
#     model, xt, time_steps=np.arange(1, step=1.0 / num_steps)
# )  # num_steps=num_steps)

# data = generated_datapoints.detach().cpu().numpy()
# df = pd.DataFrame(data)
# noise = pd.DataFrame(original_noise.cpu().numpy())
# df = pd.concat([df, noise], axis=1)
# df.to_csv("data.csv")
# print(data.shape)

# plt.scatter(data[:, 0], data[:, 1])
# plt.savefig(f"fig.png")

# =---------- END

# for i in range(10):
#     plt.close()
#     data = traj[(num_steps // 10) * i].detach().cpu().numpy()
#     plt.scatter(data[:, 0], data[:, 1])
#     plt.savefig(f"figs/fig_{i}.png")

# # Save generated images
# save_dir = 'test_images/'
# os.makedirs(save_dir, exist_ok=True)

# for i in range(generated_images.size(0)):
#     img = generated_images[i].detach().cpu()
#     img = (img + 1) / 2  # Scale from [-1,1] to [0,1]
#     img = torch.clamp(img, 0, 1)
#     save_image(img, os.path.join(save_dir, f'generated_image_{i+1}.png'))
