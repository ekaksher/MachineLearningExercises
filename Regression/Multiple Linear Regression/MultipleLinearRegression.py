import numpy as np
import matplotlib.pyplot as plt
import pandas

#Inserting the data set
dataset = pandas.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#Feature Scaling
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap #removing one dummmy variable.
X = X[:,1:]

#Splitting the data set for training and testing
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#Predicting the test set results.
Y_pred = regressor.predict(X_test)

#Building the optimal model with backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int),values = X , axis = 1)
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
j=regressor_OLS.summary()
print(j)
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
j=regressor_OLS.summary()
print(j)
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
j=regressor_OLS.summary()
print(j)
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
j=regressor_OLS.summary()
print(j)
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
j=regressor_OLS.summary()
print(j)