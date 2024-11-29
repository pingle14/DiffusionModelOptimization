import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data.csv")
print(data.columns)

plt.scatter(data['X1'], data['X2'])
plt.savefig(f"test_1.png")

plt.scatter(data['Z1'], data['Z2'])
plt.savefig(f"test_2.png")
