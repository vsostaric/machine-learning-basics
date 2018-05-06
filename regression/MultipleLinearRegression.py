import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("../data/50_Startups.csv");

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1:].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()

# Encode independant variable

X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])

oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
# python library actually does this automatically - done manually here for educational purposes
X = X[:,1:]

# Splitting data into Test and Training sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting the Multiple Linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)

# Predicting the test train result
y_pred = regressor.predict(X_test)

# Adding the constant factor
X = np.append(arr = np.ones(shape = (50, 1)).astype(int), values = X, axis = 1)

# Building the optimal model for Backward elimination
import statsmodels.formula.api as sm

X_opt = X[:, :]

# Fit the full model with all possible predictors
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

# See P-values of the predictors
regressor_OLS.summary()

# Remove independant variable with highest P value # repeatable
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

# Remove independant variable with highest P value # repeatable
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

# Remove independant variable with highest P value # repeatable
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

# Remove independant variable with highest P value # repeatable
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

regressor_OLS.summary()

y_pred_opt = regressor_OLS.predict(X_opt)