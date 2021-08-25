import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import Ridge

# #############################################################################
col_list = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
col_list_color = ['red','brown','green','blue','orange','pink','purple','grey']
X = pd.read_csv("data.csv", usecols = col_list)
X_scaled = X - np.mean(X) # rescale all features to have mean zero 
X_scaled = X_scaled / np.std(X_scaled) # normalize all features to var = 1 
 
target = ["Y"]
y = pd.read_csv("data.csv", usecols = target)
y = np.array(y)
y = np.reshape(y, (38,))

X_np = np.array(X_scaled)

# #############################################################################
lambdas = [0.01, 0.1, 0.5, 1, 1.5, 2.5, 10, 20, 30, 50, 100, 200, 300]

coefs = []
for l in lambdas:
    ridge = linear_model.Ridge(alpha = l, fit_intercept = True)
    ridge.fit(X_np, y)
    coefs.append(ridge.coef_)

# #############################################################################
plt.figure(figsize = (8, 6))
    
# make sure that the coefs(weights) is an array
coefs = np.array(coefs)
for col in range(coefs.shape[1]):
    plt.plot(lambdas, coefs[:,col], label = col_list[col], color = col_list_color[col])

# scale x axis to log of lambda 
plt.xscale('log') 
# legend, title, axes labels 
plt.legend(bbox_to_anchor = (1.3, 0.8))
plt.title('Coefficient Weight as lambda Grows (ridge)')
plt.ylabel('Coefficient weight')
plt.xlabel('log(lambda)')