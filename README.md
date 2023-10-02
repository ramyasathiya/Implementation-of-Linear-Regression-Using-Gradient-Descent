# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Startv the program.
2. import numpy as np.
3. Give the header to the data.
4. Find the profit of population.
5. Plot the required graph for both for Gradient Descent Graph and Prediction Graph.
6. End the program.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: JEEVITHA S
RegisterNumber:212222100016

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Predication")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(x,y,theta)

def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=x.dot(theta)
    error=np.dot(x.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))
  return theta,J_history
  
theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")


plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color='r')
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*1000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*1000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))

*/
```

## Output:

![230386308-e04bfb79-b231-453b-953e-e45512f79148](https://github.com/Jeevithha/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123623197/efbcc813-369c-4942-93dd-8cf284c023a5)

![230386410-d4ccb116-c4d8-4c4b-b348-f5ccba787338](https://github.com/Jeevithha/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123623197/99779ca9-32ff-4301-a38c-cdd8cf3e3df1)

![230386510-63de0d84-f31d-4a1c-a9fd-4972f86cf64e](https://github.com/Jeevithha/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123623197/3cb365b5-1f9d-4519-a313-31f8d0aa8eda)

![230386833-3d102068-46b6-479f-83c7-cb87f732526e](https://github.com/Jeevithha/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123623197/2a0ba535-3115-4f8e-85e0-83925aa1f72a)

![230389941-e78316c2-0ef7-40aa-8a98-036822924a2b](https://github.com/Jeevithha/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123623197/8f1e5268-e5e8-4574-afdb-3664aead3d29)

![230390024-44c07657-bcc7-42d7-a710-b70cc6d5917b](https://github.com/Jeevithha/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123623197/ef8cd857-7fc9-4018-8a74-70317dd0b078)

![230390104-41a9a384-10fe-4380-ba90-3d133ea40167](https://github.com/Jeevithha/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/123623197/45965a62-feb9-47cb-9fd9-8e244f03f503)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
