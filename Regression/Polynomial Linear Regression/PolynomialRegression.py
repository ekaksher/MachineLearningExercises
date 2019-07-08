import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

#Fitting Linear Regression To Dataset
from sklearn.linear_model import LinearRegression
linregressor_1 = LinearRegression()
linregressor_1.fit(X,Y)

#Fitting Polynomial Regression To Dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, Y)
linregressor_2 = LinearRegression()
linregressor_2.fit(X_poly, Y)

#Visualising the Linear Regression Results
plt.scatter(X,Y,color='red')
plt.plot(X,linregressor_1.predict(X),color='blue')
plt.title("Truth or Bluff(Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
#Visualising the Polynomial Regression Results
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,color='red')
plt.plot(X_grid,linregressor_2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title("Truth or Bluff(Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
 
#Prediciting a new result with Linear Regression
print(linregressor_1.predict(6.5))

#Prediciting a new result with Polynomial Regression
print(linregressor_2.predict(poly_reg.fit_transform(6.5)))