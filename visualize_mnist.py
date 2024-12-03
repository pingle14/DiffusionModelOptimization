import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages

MNIST_LEN = 28

# Ran this to fix the column headers
def fix_mnist(path="fulldataset.csv"):
    df = pd.read_csv(path)
    df.columns = (
        ["Iteration"]
        + [f"Z{i+1}" for i in range(MNIST_LEN * MNIST_LEN)]
        + [f"X{i+1}" for i in range(MNIST_LEN * MNIST_LEN)]
    )
    df.drop(columns=["Iteration"], inplace=True)
    df.to_csv(path, index=False)
    return df


def show_pics(path="fulldataset.csv", num=10):
    df = pd.read_csv(path, nrows=num)  # Read only the first 'num' rows
    figs = []

    for i in range(num):
        # Extract the pixel data for each image (assuming it's flattened)
        data = pd.DataFrame()
        data["data"] = np.array(df.iloc[i])
        pic = data[-784:]  # Get the last 784 columns, representing one 28x28 image
        pic = np.array(pic)

        # Reshape the 1D array into a 28x28 2D array
        image_array = pic.reshape(28, 28)

        # Normalize the pixel values to the range [0, 255]
        # If the values are already between 0 and 255, this step is unnecessary
        # If they are between 0 and 1, multiply by 255
        normalized_image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)

        # Convert the 2D array into a PIL image (grayscale mode 'L')
        image = Image.fromarray(normalized_image_array.astype(np.uint8), mode="L")

        # Create the plot
        fig, ax = plt.subplots()
        plt.imshow(image, cmap="gray")  # 'gray' colormap for grayscale images
        plt.axis("off")  # Turn off the axis
        figs.append(fig)

    # Save all figures into a single PDF
    with PdfPages(f"output_plots_n{num}.pdf") as pdf:
        for fig in figs:
            pdf.savefig(fig)  # Save the figure to the PDF
            plt.close(fig)  # Close the figure to free memory

show_pics()
