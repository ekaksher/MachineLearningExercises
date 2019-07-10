#Decision Tree Regression Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset.
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2:].values

#Fitting the Decision Tree Regression To The Dataset.
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,Y)
#Predicting a new Result
Y_pred = regressor.predict(6.5)

#Visualising the Decision Tree Regression Results
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title("Truth or Bluff (Decison Tree Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Visualising the Decision Tree Regression Results in Higher Resolution
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title("Truth or Bluff (Decison Tree Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
