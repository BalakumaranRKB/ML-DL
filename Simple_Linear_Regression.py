# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 21:23:28 2021

@author: Kumaran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\Kumar\Desktop\DS+ML Practice\Machine Learning Playlist\simple-Linear-Regression-master\Electricity_Consumption.csv')
X = dataset.iloc[0:,2].values
y = dataset.iloc[0:,1]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3,random_state = 0)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'purple')
plt.title('Electricity_Consumption vs Size_of_House(Test_Set)')
plt.xlabel('Size_of_House')
plt.ylabel('Electricity_Consumption')
plt.show()
