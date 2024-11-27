import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data.csv")
print(data.columns)

plt.scatter(data['0'], data['1'])
plt.savefig(f"test_1.png")

plt.scatter(data['0.1'], data['1.1'])
plt.savefig(f"test_2.png")
