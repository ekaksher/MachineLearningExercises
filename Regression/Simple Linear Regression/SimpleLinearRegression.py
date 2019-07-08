#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset=pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predecting the Test Set Results
Y_pred = regressor.predict(X_test)

#Visulaising the Training set results.
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()

#Visulaising the Test set results.
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()

