# LASSO, Ridge Regression and Leave One Out Cross Validation
This repository analyses the differences between LASSO and Ridge Regression and how we can use Leave One Out Cross Validation (LOOCV) to evaluate the performance of these models. The LOOCV implementation does not use any existing packages that implemenet cross validation and is written completely from scratch. 


## Part 1 - Exploratory Data Analysis 
By using a pairs plot, we can study the correlations between the features. 

![image](https://user-images.githubusercontent.com/43845085/131055946-56f187db-1a46-4955-b4fd-d80beacabea4.png)

Frome the pairs plot above, we can see almost perfect correlation between X3, X4 and X5. It can be inferred that there is a large amount of multicollinearity in this dataset. If we include all of the variables into a model, we will introduce multicollinearity into the regression. This is an issue as we will not be able to compute the MLE estimates of the coefficients which subsequently effects the interpretability of our model.

## Part 2 - Pre Processing 
In order for LASSO and Ridge to be run properly, we often rescale the features in the dataset. In our dataset, we need to rescale each feature so that it has zero mean. We rescale it again such that the sum of squared observations of each of the 8 transformed features is equal to the number of observations (in our case there are 38 observations).

 ![image](https://user-images.githubusercontent.com/43845085/131056044-f8f9dee1-db72-49a1-83ea-cdad4ec89556.png)

## Part 3 - Ridge Regression Implementation 
By running ridge regression, we can see that as the penalty value lambda increases, the coefficients of the features shrink towards zero. Features X3 and X4 seem to be symmetric at first with the weight of X4 becoming more negative as the weight of X3 becomes more positive. They then converge on the same path towards zero. X4 and X5 go towards each other at first, crossing paths until they eventually go towards convergence. X1 seems to be the most significant. 

 ![image](https://user-images.githubusercontent.com/43845085/131056087-44d98974-fe5c-4888-880d-4b8da7afd22b.png)

## Part 4 - Ridge LOOCV Implementation (from scratch) 
In this section, we will find the best value of lambda for the ridge regression problem. We will then compare the results to standard ordinary least squares (OLS) in order to determine which gives the better prediction error. 

The best lambda value was when lambda = 22.3 which produced a LOOCV error of approximately 1442.698. Obviously, this is doing much better than the OLS case since OLS corresponds to the lambda = 0 case.  

 ![image](https://user-images.githubusercontent.com/43845085/131056412-04605794-b922-41a1-8729-866e8d609fca.png)

## Part 5 - LASSO Regression Implementation 
We see a similar pattern to the ridge case except that now the 1-norm penalty of the LASSO forces X4 to be zero and attached all weight to X3 since these two variables carry the same signal. In the ridge case the signal was spread uniformly across, but the LASSO prefers sparse solutions and so picks only one of them. The LASSO ends up setting all variables to zero as λ is made to be increasingly large, something that does not happen in the Ridge case. 

 ![image](https://user-images.githubusercontent.com/43845085/131056448-a0a769eb-48b5-416b-829b-97ebae7785ab.png)

## Part 6 - LASSO LOOCV Implementation (from scratch) 
The best lambda value was at lambda = 5.5 which produced a LOOCV error of approximately 1586.67. 

 ![image](https://user-images.githubusercontent.com/43845085/131056501-4e1f3588-b119-4e8b-91fd-5527b897ce94.png)
 
 ## Discussion (LASSO vs Ridge)
In terms of error, the use of ridge regression provided a smoother error curve compared to lasso regression. Ridge regression in this case also presented us with lower error compared to the lasso regression. Although lasso has the ability to help with variable selection due to its ability to set some coefficients to zero, I personally prefer ridge regression as using the L2-Norm is more consistent with statistical analysis compared to the L1-Norm. Moreover, since the coefficients are squared in the penalty expression when using L2-Norm, it forces the coefficient values to be spread out more equally whereas L1-Norm does not. Therefore, since ridge regression provides models that are generally more stable as the coefficients do not fluctuate on small data changes, it is more preferable for feature interpretation. 


