import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import csv


class ToyDataset(Dataset):
    def __init__(self, n_samples=1000, noise=0.1, dimension=2, max_t=4 * np.pi):
        """
        Args:
            n_samples (int): Number of samples in the dataset.
            noise (float): Amount of noise added to the spiral.
            dimension (int): The dimensionality of the data (2D or higher).
            max_t (float): Maximum value for the parameter t.
        """
        self.n_samples = n_samples
        self.noise = noise
        self.dimension = dimension
        self.max_t = max_t

        # Generate the parameter t and the radial distance (r)
        self.t = np.linspace(0, self.max_t, self.n_samples)
        # Adding noise to the radius
        self.r = self.t + self.noise * np.random.randn(self.n_samples)

        # Generate the spiral data
        self.X = self.generate_spiral_data()

        # If the data is 2D, plot it
        if self.dimension == 2:
            plt.scatter(self.X[:, 0], self.X[:, 1])
            plt.savefig("plots/input_data.png")

    def generate_spiral_data(self):
        """
        Generate spiral data for the given number of dimensions.
        """
        data = np.zeros((self.n_samples, self.dimension))

        # Generate the 2D spiral using cos and sin for the first two dimensions
        data[:, 0] = self.r * np.cos(self.t)  # X-coordinate
        data[:, 1] = self.r * np.sin(self.t)  # Y-coordinate

        if self.dimension > 2:
            # For higher dimensions, we rotate the spiral into the additional dimensions
            # Use different frequencies for each additional dimension
            for i in range(2, self.dimension):
                data[:, i] = self.r * np.cos((i - 1) * self.t)
        return data

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Return the data point and its corresponding parameter t value
        return torch.tensor(
            self.X[idx], dtype=torch.float32
        )  # , torch.tensor(self.t[idx], dtype=torch.float32)


# DataModule to handle data loading
class ToyDataModule(pl.LightningDataModule):
    def __init__(
        self, csv_file, batch_size=128, n_samples=10000, noise=0.1, dimension=2
    ):
        super().__init__()
        ### TODO: sample datapoints from the toy function, either pre-computed, or create them here
        # self.csv_file = csv_file
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.noise = noise
        self.dimension = dimension

    def setup(self, stage=None):
        # df = pd.read_csv(self.csv_file)
        self.dataset = ToyDataset(
            n_samples=self.n_samples, noise=self.noise, dimension=self.dimension
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )


def save_dataset_to_csv(dataset, filename="spiral_data.csv"):
    # Open a CSV file to write the data
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        # Write the header: 'feature_1', 'feature_2', ..., 'feature_n', 't'
        header = [f"feature_{i+1}" for i in range(dataset.dimension)] + ["t"]
        writer.writerow(header)

        # Write data rows
        for i in range(len(dataset)):
            data_point = dataset[i].numpy()  # Convert torch tensor to numpy array
            t_value = dataset.t[i]  # Extract the t value from the dataset
            writer.writerow(list(data_point) + [t_value])


def generate_saved_data(n_samples=10, dims=2, path=f"data/"):
    dataset = ToyDataset(n_samples=n_samples, noise=0.5, dimension=dims)
    save_dataset_to_csv(dataset, f"{path}spiral_n{n_samples}_d{dims}.csv")
