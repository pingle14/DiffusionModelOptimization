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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import MNIST
import numpy as np
import pandas as pd
import pytorch_lightning as pl

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
        return input_noise, x_i, x_i_store
  
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
        return self.ddpm_model.sample(x_i, c_i, context_mask, self.size, self.device, guide_w=0.0, enable_store=True)
    
    
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
    trainer = pl.Trainer(accelerator="gpu", devices="auto")  # Adjust 'devices' to the number of GPUs

    # Run sampling on multiple GPUs
    #input_noise, x_gen, _ 
    stuff = trainer.predict(lit_sampler, sampling_dataloader, return_predictions=True)
    print(type(stuff), type(stuff[0]), type(stuff[0][0]), type(stuff[0][0][0]))
    input_noise = stuff[0]
    x_gen = stuff[1]

    noise = pd.DataFrame(input_noise)
    gen = pd.DataFrame(x_gen)
    
    df = pd.concat([noise, gen], axis=1)
    df.columns = [f'Z_{i}' for i in range(input_noise.shape[1])] + [f'X_{i}' for i in range(input_noise.shape[1])]
    df.to_csv('NEW_mnist_inferences.csv', index=False)

    # Use the sampling function directly
    #results = lit_sampler.sample(n_sample, size, guide_w=0.5)

    """ddpm_module = DDPM_Lightning_Module(ddpm)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',  # Updated precision
        num_nodes=1,
    )
    trainer.test(ddpm_module)"""
    """ep = "test"
    ddpm.eval()
    print(len(dataset))
    with torch.no_grad():
        n_sample = 10000
        # disable in sample the code that saves stuff to x_gen_store!!!!
        input_noise, x_gen, _ = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=guidance_wieght, enable_store=False)
        # and then flatten x_gen, so that its dimensions are n_sample x (everything else)
        # and then save to a csv.
        x_gen_data = x_gen.flatten(start_dim=1)
        print(x_gen_data.shape)

        x_all = x_gen
        grid = make_grid(x_all*-1 + 1, nrow=10)
        save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
        print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")
        
        # create gif of images evolving over time, based on x_gen_store
        fig, axs = plt.subplots(nrows=int(n_sample/n_classes), ncols=n_classes,sharex=True,sharey=True,figsize=(8,3))
        def animate_diff(i, x_gen_store):
            print(f'gif animating frame {i} of {x_gen_store.shape[0]}', end='\r')
            plots = []
            for row in range(int(n_sample/n_classes)):
                for col in range(n_classes):
                    axs[row, col].clear()
                    axs[row, col].set_xticks([])
                    axs[row, col].set_yticks([])
                    # plots.append(axs[row, col].imshow(x_gen_store[i,(row*n_classes)+col,0],cmap='gray'))
                    plots.append(axs[row, col].imshow(-x_gen_store[i,(row*n_classes)+col,0],cmap='gray',vmin=(-x_gen_store[i]).min(), vmax=(-x_gen_store[i]).max()))
            return plots
        ani = FuncAnimation(fig, animate_diff, fargs=[x_gen_store],  interval=200, blit=False, repeat=True, frames=x_gen_store.shape[0])    
        ani.save(save_dir + f"gif_ep{ep}_w{w}.gif", dpi=100, writer=PillowWriter(fps=5))
        print('saved image at ' + save_dir + f"gif_ep{ep}_w{w}.gif")"""


if __name__ == "__main__":
    test_mnist("./")
    pass
