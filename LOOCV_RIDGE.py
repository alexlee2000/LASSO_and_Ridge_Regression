import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
import numpy as np # to create the lambda grid 
from sklearn import linear_model
from sklearn.linear_model import Ridge # for ridge regression only 

# #############################################################################
col_list = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
X = pd.read_csv("data.csv", usecols = col_list)
X_scaled = X - np.mean(X) # rescale all features to have mean zero 
X_scaled = X_scaled / np.std(X_scaled) # normalize all features to var = 1 
target = ["Y"]
y = pd.read_csv("data.csv", usecols = target)
y = np.array(y)
y = np.reshape(y, (38,))
X_np = np.array(X_scaled)

# #############################################################################
lambdas = np.arange (0, 50.1, 0.1) # 0, 0.1, 0.2, ..., 50
loo_err_avg = [] # initialise the Leave-one-out error array used to average the LOO for all n for all lambda 
loo_counter = 0 # index for the loo_err_avg arrary above
lambda_optimal = 0
loo_err_avg_optimal = 0 

for l in lambdas:
    loo_err = [] # initialise the Leave-one-out error used to calculate n errors of a given lambda.
    
    for n in list(range(0,38)): # index 0, ..., 37  

        X_train = X_np # copy X_np into X_train 
        X_train = np.delete(X_train, n, 0) # every row except n 
        X_test = X_np[n] # row n 
    
        y_train = y 
        y_train = np.delete(y_train, n, 0) 
        y_test = y[n] 

        # ridge regression on the train set 
        ridge_weight = []
        ridge = Ridge(alpha = l, fit_intercept = True)
        ridge.fit(X_train, y_train)
        ridge_weight.append(ridge.coef_)
        
        # test the ridge on the test set (find the error and save into loo_err)
        y_pred = ridge.predict(X_test.reshape(1,-1))
        loo_err.append(np.sum((y_test - y_pred)**2))

    loo_err_avg.append(np.mean(loo_err))
    if (min(loo_err_avg) == loo_err_avg[loo_counter]):
        lambda_optimal = l
        loo_err_avg_optimal = loo_err_avg[loo_counter]

    loo_counter = loo_counter + 1
# #############################################################################
plt.plot(lambdas, loo_err_avg)
plt.title('Leave-One-Out Error as lambda grows')
plt.ylabel('LOO Error')
plt.xlabel('lambda')
print("The best lambda value = ", lambda_optimal)
print("The LOO Error at lambda ", lambda_optimal, " is ", loo_err_avg_optimal)

lregr = linear_model.LinearRegression()
lregr.fit(X,y)
predicted_y = lregr.predict(X)
OLS_Error = mean_squared_error(y,predicted_y)
print(OLS_Error)
