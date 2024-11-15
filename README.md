# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import pandas, numpy and mathplotlib.pyplot.

2.Trace the best fit line and calculate the cost function.

3.Calculate the gradient descent and plot the graph for it.

4.Predict the profit for two population sizes.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Ramya S
RegisterNumber:  212222040130
```
```


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
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
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
*/
```

## Output:

![image](https://github.com/user-attachments/assets/bcab0011-be4c-4fc8-b8e8-a139be50fce5)
```
![image](https://github.com/user-attachments/assets/08f0b8cc-87e6-49a3-95e4-82fb96748087)
```
![image](https://github.com/user-attachments/assets/c37e448b-bc44-4a17-84a5-2407509763f0)
```
![image](https://github.com/user-attachments/assets/063d43dd-4cb4-409c-a04a-4e37aad46482)
```
![image](https://github.com/user-attachments/assets/a18b426d-7892-40f5-9cb5-f5c5e9663228)







## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
