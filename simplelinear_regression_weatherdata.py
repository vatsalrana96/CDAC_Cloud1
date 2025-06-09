# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 22:29:20 2020

@author: cdac
"""


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('D:\\dataset\\weather.csv')
dataset.shape
dataset.describe()

#let’s plot our data points on a 2-D graph to eyeball our dataset
dataset.plot(x='MinTemp', y='MaxTemp', style='o')  
plt.title('MinTemp vs MaxTemp')  
plt.xlabel('MinTemp')  
plt.ylabel('MaxTemp')  
plt.show()

#the average maximum temperature
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['MaxTemp'])

X = dataset['MinTemp'].values.reshape(-1,1)
X
y = dataset['MaxTemp'].values.reshape(-1,1)

#arr=np.arange(2,50,2).reshape(-1,4)
#arr
y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=150)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm


#To retrieve the intercept:
print(regressor.intercept_)
print('Coefficients: \n', regressor.coef_)
#For retrieving the slope:
print(regressor.coef_)
y_pred = regressor.predict(X_test)

y_pred
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()


print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.metrics import r2_score
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
  
