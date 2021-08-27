import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error, r2_score

# load the dataset 
data = pd.read_csv("data.csv")
X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

sns.pairplot(data.iloc[:,:-1])
plt.savefig("figures/PairsPlot.png", dpi=400)

plt.show()

