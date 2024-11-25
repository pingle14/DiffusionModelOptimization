import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd

class ToyDataset(Dataset):
    def __init__(self, n_samples=1000, noise=0.1, dimension=2, max_t=4*np.pi):
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

        # Generate the data
        self.t = np.linspace(0, self.max_t, self.n_samples)
        self.r = self.t + self.noise * np.random.randn(self.n_samples)  # Adding noise to the radius
        
        # Generate spiral data in the specified number of dimensions
        self.X = self.generate_spiral_data(self.t, self.r, self.dimension)

    def generate_spiral_data(self, t, r, dimension):
        """
        Generate spiral data for the given number of dimensions.
        """
        data = np.zeros((self.n_samples, dimension))
        for i in range(dimension):
            # Use sine and cosine functions for the first two dimensions, then rotate for higher dims
            data[:, i] = r * np.cos((i + 1) * t)  # Shift frequency for each dimension
        return data

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Return the data point and its corresponding parameter t value
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.t[idx], dtype=torch.float32)

# DataModule to handle data loading
class ToyDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, batch_size=128, noise=0.5):
        super().__init__()
        ### TODO: sample datapoints from the toy function, either pre-computed, or create them here
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.noise = noise

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_file)
        self.dataset = ToyDataset(df, noise=self.noise, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
