import numpy as np
import pandas as pd

col_list = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
X = pd.read_csv("data.csv", usecols=col_list)

X_scaled = X - np.mean(X) # rescale all features to have mean zero 
# print(np.mean(X.X1))
X_scaled = X_scaled / np.std(X_scaled) # normalize all features to var = 1 
# print(np.mean(X.X2))

# verifies if var = 1 
# np.var(X, axis=0) 

i = 0
summ = 0
while (i < 38): # loops thru each observation 
    summ = summ + X_scaled.X1[i]**2
    i = i + 1
print( "X1: " + str(summ))

i = 0
summ = 0
while (i < 38): # loops thru each observation 
    summ = summ + X_scaled.X2[i]**2
    i = i + 1
print("X2: " + str(summ)) 

i = 0
summ = 0
while (i < 38): # loops thru each observation 
    summ = summ + X_scaled.X3[i]**2
    i = i + 1
print("X3: " + str(summ)) 

i = 0
summ = 0
while (i < 38): # loops thru each observation 
    summ = summ + X_scaled.X4[i]**2
    i = i + 1
print("X4: " + str(summ)) 

i = 0
summ = 0
while (i < 38): # loops thru each observation 
    summ = summ + X_scaled.X5[i]**2
    i = i + 1
print("X5: " + str(summ)) 

i = 0
summ = 0
while (i < 38): # loops thru each observation 
    summ = summ + X_scaled.X6[i]**2
    i = i + 1
print("X6: " + str(summ)) 

i = 0
summ = 0
while (i < 38): # loops thru each observation 
    summ = summ + X_scaled.X7[i]**2
    i = i + 1
print("X7: " + str(summ)) 

i = 0
summ = 0
while (i < 38): # loops thru each observation 
    summ = summ + X_scaled.X8[i]**2
    i = i + 1
print("X8: " + str(summ)) 


