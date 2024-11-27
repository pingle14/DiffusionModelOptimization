from diffusion_model.dataset import ToyDataset, generate_saved_data
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from diffusion_model.sampler import euler_sampler
import torch.nn.functional as F


# Define the Neural Network model
class GaussianNoiseNN(nn.Module):
    def __init__(self, input_size, output_nTimesteps):
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
        adjusted_mse = F.mse_loss(output_generation, target_generation)

        # Final custom loss: combine all components with respective weights
        loss = self.alpha * distinctness_loss + self.beta * adjusted_mse + range_loss
        return loss


def READ_BATCH():
    raise NotImplementedError


# Hyperparameters
input_size = 1  # Scalar input (e.g., a random noise value)
hidden_size = 64
output_nTimesteps = 5  # Output per timestep
learning_rate = 0.001
epochs = 1000
device = "cpu"

# Initialize the neural network
model = GaussianNoiseNN(input_size, output_nTimesteps)

# Loss function (Mean Squared Error for regression)
criterion = TimestepLoss(alpha=0.0, beta=1.0)

# Optimizer (Adam optimizer)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# TODO: Train-Test-Split on CSV

# Training loop
for epoch in range(epochs):
    # Generate input (random Gaussian noise)
    noise, target_generations = READ_BATCH()

    # Forward pass
    output_timesteps = model(noise)
    output_generation = euler_sampler(
        model, xt=noise, time_steps=output_timesteps, device=device
    )

    # Compute the loss
    loss = criterion(output_timesteps, output_generation, target_generations)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# TODO: Test the model after training
# Create GEneration plot
