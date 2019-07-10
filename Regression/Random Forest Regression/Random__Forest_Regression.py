import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:].values

#Predicting a new Result
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300,random_state=0)
regressor.fit(X,Y)

#predicting a new result
y_pred = regressor.predict(6.5)

#Visualising the Regression Results in A higher Resolution
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title("Truth or Bluff (Random Forest Regression)")
plt.xlabel("Postion Level")
plt.ylabel("Salary")
plt.show()