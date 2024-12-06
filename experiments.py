from diffusion_model.dataset import ToyDataset, generate_saved_data
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed
from diffusion_model.sampler import euler_sampler, mnist_sampler
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
from diffusion_model.spiral_model import SpiralDiffusionModel
from diffusion_model.mnist_model import LitSampler, DDPM, ContextUnet
import argparse
#import traceback
#import warnings

# Ignore all warnings
#warnings.filterwarnings("ignore")

sampler_fns = {'spiral':euler_sampler, 'mnist': mnist_sampler}

# Define the Neural Network model
# Input = Noise, Output = Timesteps (k)
# TODO: Add class label here?
class GaussianNoiseNN(nn.Module):
    def __init__(self, input_size=2, output_ntimejumps=500, label_dim=0):
        super(GaussianNoiseNN, self).__init__()

        # Define the layers of the network
        input_size += label_dim
        self.fc1 = nn.Linear(input_size, 64)  # First hidden layer with 64 units
        self.fc2 = nn.Linear(64, 128)  # Second hidden layer with 128 units
        self.fc3 = nn.Linear(128, output_ntimejumps)  # Output layer with k units

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Sigmoid to constrain output between [0, 1]
        self.softmax = nn.Softmax()

    def forward(self, x, c=None):
        # print(x.shape, c.shape)
        if (c is not None):
            x = torch.cat((x, c.unsqueeze(-1)), dim=1)
        
        x = self.relu(self.fc1(x))  # Pass input through the first hidden layer
        x = self.relu(self.fc2(x))  # Pass through the second hidden layer
        # Apply sigmoid to output to ensure it's between [0, 1]
        x = self.softmax(self.fc3(x))  # Replaced sigmoid with softmax in Approach 2

        x = torch.cumsum(x, dim=1)  # Prefix Sum to convert changes to timejumps

        # Post-processing to make sure outputs are distinct
        # Normalize
        # x = self.normalize(x)

        return x

    def normalize(self, x):
        total = torch.norm(x)
        return x / total

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

    def forward(self, output_timejumps, output_generation, target_generation):
        # # Loss 1: Penalize outputs that are outside the [0, 1] range (just as a safety check)
        # range_loss = torch.sum(
        #     torch.clamp(output_timejumps, min=0.0, max=1.0) - output_timejumps
        # )

        # Loss 2: Penalize if the values are too close to each other (distinctness penalty)
        # Compute pairwise differences between outputs
        pairwise_diff = torch.triu(
            torch.abs(output_timejumps[:, None] - output_timejumps), diagonal=1
        )
        distinctness_loss = torch.sum(
            torch.exp(-pairwise_diff)
        )  # Penalize small differences

        # Loss 3: Actual Objective Function
        # output_generation = output_generation.reshape(output_generation.shape[0], -1)
        adjusted_mse = F.mse_loss(output_generation, target_generation)

        # Final custom loss: combine all components with respective weights
        loss = self.alpha * distinctness_loss + self.beta * adjusted_mse  # + range_loss
        return loss


class CSVDataset(Dataset):
    def __init__(self, df, dims=2, conditional_col='C'):
        if "Iteration" in df.columns:
            df.drop(columns=["Iteration"], inplace=True)
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)
        self.dims = dims
        self.conditional = conditional_col in df.columns
        self.input_noise = np.array(df[[f"Z{i}" for i in range(self.dims)]])
        self.target_generation = np.array(df[[f"X{i}" for i in range(self.dims)]])
        self.class_label = np.array(df["C"]) if self.conditional else None
        self.n_samples = len(self.input_noise)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Return the data point and its corresponding parameter t value
        item = {
            "input_noise": torch.tensor(self.input_noise[idx], dtype=torch.float32),
            "output_generation": torch.tensor(self.target_generation[idx], dtype=torch.float32).reshape(1, 28, 28)
        }
        if self.conditional:
            item["condition_label"] = torch.tensor(self.class_label[idx], dtype=torch.int64)
        return item


# DataModule to handle data loading
class CSVDataModule(pl.LightningDataModule):
    def __init__(
        self,
        csv_file,
        dims,
        conditional_col='C',
        batch_size=128,
        train_val_test_split=[0.8, 0.1, 0.1],
    ):
        super().__init__()
        ### TODO: sample datapoints from the toy function, either pre-computed, or create them here
        self.batch_size = batch_size
        self.dims = dims
        self.conditional_col = conditional_col
        self.df = pd.read_csv(csv_file)
        self.train_val_test_split = train_val_test_split

    def setup(self, stage=None):
        # df = pd.read_csv(self.csv_file)
        self.dataset = CSVDataset(df=self.df, dims=self.dims, conditional_col=self.conditional_col)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, self.train_val_test_split
        )

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
    def __init__(
        self,
        diffusion_model,
        inference_results_path,
        sampler_fn,
        label_dim=0,
        input_size=2,
        output_ntimejumps=20,
        loss_fn=TimestepLoss(alpha=0.0, beta=1.0),
        learning_rate=1e-5,
    ):
        super(TimseStepSelectorModule, self).__init__()
        self.model = GaussianNoiseNN(
            input_size=input_size, output_ntimejumps=output_ntimejumps, label_dim=label_dim
        )
        self.inference_results_path = inference_results_path
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.train_loss = MeanMetric()
        self.validation_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.scaler = GradScaler()
        self.diffusion_model = diffusion_model
        self.inference_results = []
        self.sampler_fn = sampler_fn
        for param in self.diffusion_model.parameters():
            param.requires_grad = False

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        with autocast(device_type="cuda"):
            input_noise = batch["input_noise"]
            target_generations = batch["output_generation"]
            condition_label = batch.get('condition_label', None)
            output_timejumps = self.model(input_noise, condition_label)
            output_generations = self.sampler_fn(
                self.diffusion_model,
                xt=input_noise,
                time_jumps=output_timejumps,
                device=device,
                c_t=condition_label,
            )
            loss = self.loss_fn(
                output_timejumps, output_generations, target_generations
            )

        # Log the loss
        self.train_loss.update(loss)
        self.log("train_loss", self.train_loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        with autocast(device_type="cuda"):
            input_noise = batch["input_noise"]
            target_generations = batch["output_generation"]
            condition_label = batch.get("condition_label", None)
            output_timejumps = self.model(input_noise, condition_label)
            output_generations = self.sampler_fn(
                self.diffusion_model,
                xt=input_noise,
                time_jumps=output_timejumps,
                device=device,
                c_t=condition_label,
            )
            loss = self.loss_fn(
                output_timejumps, output_generations, target_generations
            )

        # Log the loss
        self.validation_loss.update(loss)
        self.log("validation_loss", self.validation_loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        with autocast(device_type="cuda"):
            input_noise = batch["input_noise"]
            target_generations = batch["output_generation"]
            condition_label = batch.get("condition_label", None)
            output_timejumps = self.model(input_noise, condition_label)
            output_generations = self.sampler_fn(
                self.diffusion_model,
                xt=input_noise,
                time_jumps=output_timejumps,
                device=device,
                c_t=condition_label,
            )
            loss = self.loss_fn(
                output_timejumps, output_generations, target_generations
            )

        # Log the loss
        self.test_loss.update(loss)
        self.log("test_loss", self.test_loss, prog_bar=True)
        # Collect the data from the current batch
        batch_size = input_noise.size(0)

        for i in range(batch_size):
            # Flattening each of the tensors to single lists of values
            # Input tensor as numpy
            input_data = input_noise[i].cpu().numpy()
            # Predicted tensor as numpy
            predicted_data = output_generations[i].cpu().numpy()
            # Target tensor as numpy
            target_data = target_generations[i].cpu().numpy()

            item = {
                **{
                    f"input_{j}": input_data[j] for j in range(len(input_data))
                },  # Each element in input_noise
                **{
                    f"predicted_{j}": predicted_data[j]
                    for j in range(len(predicted_data))
                },  # Each element in output_generations
                **{
                    f"target_{j}": target_data[j] for j in range(len(target_data))
                },  # Each element in target_generations
            }

            if 'condition_label' in batch:
                condition_data = condition_label[i].cpu().numpy()
                item["condition_label"] = condition_data

            # Add each example's data to the results list
            self.inference_results.append(item)

        return loss

    def on_test_end(self):
        # After the test loop is done, save results to a CSV
        print("Saving inference results to CSV...")
        # Use pandas for easy saving to CSV
        df = pd.DataFrame(self.inference_results)
        df.to_csv(self.inference_results_path, index=False)
        # plt.scatter(df.values[:, 0], df.values[:, 1])
        # plt.savefig(f"input_noise.png")
        # plt.close()
        # plt.scatter(df.values[:, 2], df.values[:, 3])
        # plt.savefig(f"small_time_prediction.png")
        # plt.close()
        # plt.scatter(df.values[:, 4], df.values[:, 5])
        # plt.savefig(f"actual.png")
        # plt.close()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=10000, power=2
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def train_time_model(
    diffusion_model_path,
    time_model_directory,
    inference_results_path,
    sampler_fn_name,
    data_module,
    num_epochs=5000,
    num_time_jumps=20,
    num_gpus=8,
    existing_time_model=None,
    dims = 2
):

    # Instantiate the model, data module, and trainer
    diffusion_model = None
    label_dim = 0
    if sampler_fn_name == 'mnist':
        n_Timesteps = 400 # 500
        device = "cuda"
        n_classes = 10
        n_feat = 128 # 128 ok, 256 better (but slower)
        guidance_wieght = 2.0
        diffusion_model = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_Timesteps=n_Timesteps, device=device, drop_prob=0.1)
        diffusion_model.load_state_dict(torch.load(diffusion_model_path))
        diffusion_model.to(device)
        label_dim = 1
    elif sampler_fn_name == 'spiral':
        diffusion_model = SpiralDiffusionModel.load_from_checkpoint(diffusion_model_path)
    else:
        raise ValueError(f'Invalid sampler_fn_name: {sampler_fn_name}')

    
    model = existing_time_model
    if model is None:
        model = TimseStepSelectorModule(
            diffusion_model=diffusion_model,
            inference_results_path=inference_results_path,
            sampler_fn=sampler_fns[sampler_fn_name],
            output_ntimejumps=num_time_jumps,
            label_dim=label_dim,
            input_size=dims
        )
    else:
        model = TimseStepSelectorModule.load_from_checkpoint(existing_time_model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=time_model_directory,  # Directory to save the models
        filename="time-model-{epoch:02d}",  # Filename format
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=500,  # Save every 5 epochs
    )

    # Define the trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu",
        devices=num_gpus,
        precision="16-mixed",  # Updated precision
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
    )

    # Start training
    trainer.fit(model, datamodule=data_module)


def visualize_model(data_module, time_model_path, diffusion_model):
    # TODO: Ok to load from this checkpoint? Will it know the Diffusion Model?
    time_model = TimseStepSelectorModule.load_from_checkpoint(
        time_model_path, **{"diffusion_model": diffusion_model}
    )

    # Initialize the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",  # Updated precision
        num_nodes=1,
    )

    # Run the test phase
    trainer.test(time_model, datamodule=data_module)


# Main training script
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Experiment CLI args")
    # Add arguments
    parser.add_argument(
        "-m",
        "--SpiralDiffusionModel",
        action="store",
        help="Enter path where to load fully trained diffusion model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--csvFile",
        action="store",
        help="Enter CSV that stores inferences from the diffusion model. This CSV is used to train the TimeStepModel",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--outputTimeModelDirPath",
        action="store",
        help="Enter dir path where to store output trained model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-t",
        "--timeModel",
        action="store",
        help="Enter dir path to load an existing time model",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Only do visualizations",
        required=False,
    )
    parser.add_argument(
        "-n",
        "--numTimeJumps",
        default=20,
        action="store",
        help="Enter number of time steps for model to use",
        type=int,
        required=False,
    )
    parser.add_argument(
        "-d",
        "--dims",
        action="store",
        help="Enter dims in the inference dataset (for MNIST this is 784)",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-r",
        "--inferenceResultsPath",
        action="store",
        help="Enter path where to store inference CSV",
        type=str,
        required=True,
    )
    # Add argument with restricted choices
    parser.add_argument(
        "-s",
        "--sampler",
        choices=list(sampler_fns.keys()),
        help=f"Choose a dataset sampler: {list(sampler_fns.keys())}",
        required=True
    )

    # Parse the arguments
    args = parser.parse_args()

    # CSV: "diffusion_model/data.csv"
    data_module = CSVDataModule(csv_file=args.csvFile, dims=args.dims, batch_size=8)

    if args.visualize:
        visualize_model(
            data_module=data_module,
            time_model_path=args.timeModel,
            diffusion_model=SpiralDiffusionModel.load_from_checkpoint(args.SpiralDiffusionModel),
        )
        exit()

    train_time_model(
        # "model_files6/toy_model-epoch=1999.ckpt"
        diffusion_model_path=args.SpiralDiffusionModel,
        # "time_model_files/"
        time_model_directory=args.outputTimeModelDirPath,
        inference_results_path=args.inferenceResultsPath,
        sampler_fn_name=args.sampler,
        data_module=data_module,
        num_epochs=5000,
        num_time_jumps=args.numTimeJumps,
        num_gpus=8,
        existing_time_model=args.timeModel,
        dims=args.dims
    )
    #except Exception as e:
    #    if torch.distributed.get_rank() == 0:  # Only rank 0 prints the error
    #        print(f"Error occurred: {e}")
    #        traceback.print_exc()
