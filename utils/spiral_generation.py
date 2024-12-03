import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Function to generate a 2D Swiss roll
def generate_2d_swiss_roll(n_samples=1000, noise=0.1, visualize=False):
    # Generate a parameter t (from 0 to 4*pi, controlling how tightly the roll is)
    t = np.linspace(0, 4 * np.pi, n_samples)

    # Generate radius as a function of t (the "rolling out" effect)
    r = t + noise * np.random.randn(n_samples)  # Radius increases with t and some noise

    # Parametric equations for the 2D Swiss roll
    x = r * np.cos(t)  # x-coordinate (based on polar coordinates)
    y = r * np.sin(t)  # y-coordinate

    data = np.column_stack((x, y))

    if visualize:
        # Visualize the 2D Swiss roll
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            data[:, 0], data[:, 1], c=t, cmap="viridis", edgecolors="k", s=50
        )
        plt.title("2D Swiss Roll")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar(scatter, label="Parameter t")
        plt.savefig("plots/2DSwissRoll.png")

    return (data, t)


# def generate_2d_swiss_roll(n_samples=1000, visualize=False):
#     X, color = make_swiss_roll(n_samples)

#     # Discard the z-dimension to make it 2D (use only x and y)
#     X_2d = X[:, :2]  # Only keep the first two dimensions (x and y)
#     if visualize:
#         # Visualize the 2D Swiss roll
#         plt.figure(figsize=(10, 7))
#         scatter = plt.scatter(
#             X_2d[:, 0], X_2d[:, 1], c=color, cmap="cividis", edgecolors="k", s=50
#         )
#         plt.title("2D Swiss Roll (Projection of 3D Swiss Roll)")
#         plt.xlabel("X")
#         plt.ylabel("Y")
#         plt.colorbar(scatter, label="Parameter (Color based on z)")

#         # Show the plot
#         plt.savefig("plots/2DSwissRoll.png")

#     return X_2d


# Function to create a 10-dimensional Swiss roll
def generate_10d_swiss_roll(n_samples=1000, noise=0.1):
    # Generate the 3D Swiss roll
    X, color = make_swiss_roll(n_samples, noise=noise)

    # Parametrize the Swiss roll (unwrap the 1D curve and extend into higher dimensions)
    theta = np.arctan2(X[:, 1], X[:, 0])  # Angle in the XY plane
    z = X[:, 2]  # Height (Z axis) of the Swiss roll

    # Create a 10D Swiss roll using the original 3D structure and the parameter theta
    # Use the parametric equation to map the 1D manifold into higher dimensions
    X_10d = np.zeros((n_samples, 10))

    # First 3 dimensions are based on the 3D Swiss roll
    X_10d[:, 0] = X[:, 0]  # x dimension from Swiss roll
    X_10d[:, 1] = X[:, 1]  # y dimension from Swiss roll
    X_10d[:, 2] = X[:, 2]  # z dimension from Swiss roll

    # Map the angle 'theta' to the remaining dimensions (preserving the Swiss roll structure)
    for i in range(3, 10):
        X_10d[:, i] = (
            np.sin((i - 2) * theta) * z
        )  # This creates a "twisting" pattern across the higher dimensions

    return X_10d, color


def visualize_1st3(X_10d, color):
    # Visualize the first 3 dimensions of the Swiss roll (just for visualization purposes)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Use a darker colormap like 'cividis', 'inferno', or 'viridis'
    scatter = ax.scatter(X_10d[:, 0], X_10d[:, 1], X_10d[:, 2], c=color, cmap="cividis")

    # Add the color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label("z val", rotation=270, labelpad=20)

    # Set title and labels
    ax.set_title("3D Projection of 10D Swiss Roll")

    # Save the plot to a file
    plt.savefig("plots/10d_swiss_roll_projection.png", dpi=300)


def pca_analysis(X_10d, n_cmp=3):
    # Apply PCA to reduce 10D data to 3D for visualization
    pca = PCA(n_components=n_cmp)
    X_nd_pca = pca.fit_transform(X_10d)

    # Visualize the 3D projection of the PCA-reduced 10D data

    assert n_cmp <= 3, f"Cant visualize higher dims"
    if n_cmp == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection=f"3d")
        scatter = ax.scatter(
            X_nd_pca[:, 0], X_nd_pca[:, 1], X_nd_pca[:, 2], c=color, cmap="cividis"
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = plt.scatter(X_nd_pca[:, 0], X_nd_pca[:, 1], c=color, cmap="cividis")

    ax.set_title(f"{n_cmp}D PCA Projection of 10D Swiss Roll")
    plt.colorbar(scatter)
    plt.savefig(f"plots/pca_transform_dims{n_cmp}.png", dpi=300)


# Generate a 10D Swiss roll
n_samples = 1000  # Number of samples
X_10d, color = generate_10d_swiss_roll(n_samples)
visualize_1st3(X_10d, color)
pca_analysis(X_10d, 2)
pca_analysis(X_10d, 3)
generate_2d_swiss_roll(noise=0.5, visualize=True)
