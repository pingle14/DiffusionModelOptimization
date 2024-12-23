import os
import torch
import torch.distributed
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
import torch.optim as optim
from torchmetrics import MeanMetric
from torch.amp import autocast, GradScaler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# rom pretrained import PretrainedConvModel
# from lightning_3 import SpiralDiffusionModel
from diffusion_model.spiral_model import SpiralDiffusionModel
import matplotlib.pyplot as plt
import pandas as pd

"""# # Load the model
checkpoint_path = "../model_files6/toy_model-epoch=9999.ckpt"  #'/scratch/aadarshnarayan/models/pokemon_model-epoch=499.ckpt'
# #'models/pokemon_model-epoch=9999.ckpt'  # Update with your actual checkpoint path
model = SpiralDiffusionModel.load_from_checkpoint(checkpoint_path)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)"""

MNIST_BETAS = 1e-4, 0.02  # Hardcoded betas
MNIST_IMG_SZ = (1, 28, 28)  # Hardcoded


def generative_denoising_timestep_order(timesteps):
    return reversed(timesteps)


def noising_timestep_order(timesteps):
    return timesteps


def euler_sample_step(model, xt, i, time_jumps, device='cuda'):
    # Time variable t decreases from 1 to 0
    num_time_jumps = time_jumps.shape[1]
    step_t = time_jumps[:, i].unsqueeze(-1)

    t = step_t
    #print("step t shape", step_t.shape)
    # print(t.shape) #TODO: Handle non-tensor case with : torch.full((xt.shape[0], 1), fill_value=step, device=device)
    all_ones = torch.ones_like(step_t, device=device)
    step_size = (
        time_jumps[:, i + 1].unsqueeze(-1) if i < num_time_jumps - 1 else all_ones
    ) - step_t

    v_t = model(xt, t)
    # Update xt using Euler method
    xt = xt + step_size * v_t
    return xt


# Euler sampling function
def euler_sampler(
    model,
    xt,
    time_jumps=[],
    device="cuda",
    guide_w=None,
    c_t=None,
):
    xtraj = [xt.clone()]

    with torch.no_grad():
        num_time_jumps = time_jumps.shape[1]
        for i in range(num_time_jumps):
            xt = euler_sample_step(model, xt, i, time_jumps, device)
            xtraj.append(xt.clone().detach())

    return xt


def ddpm_schedules_nonuniform(beta1, beta2, time_jumps):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    # Ensure time_jumps is a tensor
    # time_jumps = torch.tensor(time_jumps, dtype=torch.float32)

    # Normalize the time_jumps to the range [0, 1]
    # Assuming time_jumps are in a range [0, T] where T is the largest value in time_jumps.
    T = time_jumps.max()

    # Calculate beta_t for each time step
    beta_t = (beta2 - beta1) * time_jumps / T + beta1
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


"""
output_generations = self.sampler_fn(
                self.diffusion_model,
                xt=input_noise,
                time_jumps=output_timejumps,
                device=device,
                c_t=condition_label,
                size=None,
            )

"""


def mnist_sampler(model, xt, time_jumps=[], device="cuda", guide_w=2.0, c_t=None):
    with torch.no_grad():
        n_sample = xt.shape[0]
        num_time_jumps = time_jumps.shape[1]
        beta1, beta2 = MNIST_BETAS  # Hardcoded betas
        size = MNIST_IMG_SZ  # Hardcoded
        time_jumps = time_jumps.flip(dims=[1])
        # reversed(time_jumps) #TODO: CHeck
        parameters = ddpm_schedules_nonuniform(beta1, beta2, time_jumps)

        """
        c_i (context) and context_mask are repeated twice. 
        The second half of the batch is set to context-free by setting the second half of context_mask to 1.
        TEHCNIQUE: classifier-free guidance: half of the batch is conditioned and the other half is not
        """
        context_mask = torch.zeros_like(c_t).to(device)
        c_t = c_t.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1.0  # makes second half of batch context free
        xt = xt.view(xt.shape[0], 1, 28, 28)
        # MNIST TIMESTEPS are backwards!: self.n_Timesteps, 0, -
        for i in range(num_time_jumps):
            # NOTE: when we use our own timesteps, we simply modify this line here, to use our generated timesteps
            # NOTE: Default: [i / self.n_Timesteps]
            t_is = time_jumps[:, i].view(time_jumps.shape[0], 1, 1, 1)
            # t_is = t_is.repeat(1, 1, 1, 1)

            # double batch
            xt = xt.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            # Brownian motion
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            vt_eps = model.nn_model(xt, c_t, t_is, context_mask)

            vt_eps1 = vt_eps[:n_sample]
            vt_eps2 = vt_eps[n_sample:]
            # Classifier-Free guidance:
            vt_eps = (1 + guide_w) * vt_eps1 - guide_w * vt_eps2
            xt = xt[:n_sample]

            # 125, 1, 28, 28
            # parameters : 125, 1 -> 125, 1, 1, 1
            xt = (
                parameters["oneover_sqrta"][:, i].view(-1, 1, 1, 1)
                * (xt - vt_eps * parameters["mab_over_sqrtmab"][:, i].view(-1, 1, 1, 1))
                + parameters["sqrt_beta_t"][:, i].view(-1, 1, 1, 1) * z
            )
            #print(type(xt))

    return xt


# # # Generate images
"""num_samples = 10000  # Adjust as needed
num_steps = 1000

xt = torch.randn((num_samples, 2), device=device)
original_noise = xt.clone().detach()
time_jumps = torch.tensor(np.arange(1, step=1.0 / num_steps) + np.zeros(num_samples)[:, None], device=device, dtype=torch.float)
generated_datapoints, traj = euler_sampler(
    model, xt, time_jumps=time_jumps
)  # num_steps=num_steps)

data = generated_datapoints.detach().cpu().numpy()
df = pd.DataFrame(data)
noise = pd.DataFrame(original_noise.cpu().numpy())
df = pd.concat([df, noise], axis=1)
df.to_csv("data.csv")
# print(data.shape)

plt.scatter(data[:, 0], data[:, 1])
plt.savefig(f"fig.png")

for i in range(10):
    plt.close()
    data = traj[(num_steps // 10) * i].detach().cpu().numpy()
    plt.scatter(data[:, 0], data[:, 1])
    plt.savefig(f"figs/fig_{i}.png")"""
