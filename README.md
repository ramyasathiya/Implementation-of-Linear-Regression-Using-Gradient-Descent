# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize Parameters:

Start with random values for the model's parameters: the slope (m) and the intercept (b).
2.Predict Output:

For each data point (x_i, y_i), calculate the predicted value (y_hat_i) using the current values of m and b:
y_hat_i = m * x_i + b
3.Calculate Cost Function (Mean Squared Error):

Compute the difference between the predicted value and the actual value for each data point.
Square the differences and calculate the mean of these squared differences:
MSE = (1/n) * Σ(y_i - y_hat_i)^2
4.Calculate Gradients:

Compute the partial derivatives of the cost function with respect to m and b:
∂MSE/∂m = (2/n) * Σ(x_i * (y_hat_i - y_i))
∂MSE/∂b = (2/n) * Σ(y_hat_i - y_i)
5.Update Parameters:

Update the values of m and b using the gradient descent update rule:
m = m - α * ∂MSE/∂m
b = b - α * ∂MSE/∂b
Here, α is the learning rate, which controls the step size in the gradient descent.
6.Repeat Steps 2-5:

Iterate through steps 2-5 multiple times until the cost function converges to a minimum or reaches a specified number of iterations.
## Program:

```

Program to implement the linear regression using gradient descent.
Developed by: RAMYA S
RegisterNumber:  212222040130

import numpy as  np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=100):
  X=np.c_[np.ones(len(X1)),x1]
  theta=np.zeros(X.shape[1]).reshape(-1,1)

  for _ in range(num_iters):
    predictions=(X).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)        
    theta=learning=learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta

data=pd.read_csv("50_Startups.csv")
print(data.head())
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled);

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

```

## Output:
# PREDICTED VALUE:

![Screenshot 2024-08-31 140350](https://github.com/user-attachments/assets/18385998-acfe-4b9f-a34a-6db9f81e7c00)

![Screenshot 2024-08-31 140603](https://github.com/user-attachments/assets/5d8fadbb-67e8-4c39-b4b9-bf74b3b70e19)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
