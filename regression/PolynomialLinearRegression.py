import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("../data/Position_Salaries.csv");

# Matrix of features, independant variables

X = dataset.iloc[:,1:2].values

# Dependant variables

y = dataset.iloc[:, 2].values

# Comparing Linear and Polynomial regression

# Fitting Linear regression
from sklearn.linear_model import LinearRegression

lin_regressor = LinearRegression()
lin_regressor.fit(X, y)

# Fitting Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
pol_regressor = PolynomialFeatures(degree=4)

# Transforming X into polynomial values (X^0, X^1, X^2,...)
X_poly = pol_regressor.fit_transform(X)

pol_lin_regressor = LinearRegression()
pol_lin_regressor.fit(X_poly, y)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

# Creating the plot
plt.scatter(X, y, color='red')
plt.title('Truth or bluf (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')

# Visualizing the linear regression
plt.plot(X_grid, lin_regressor.predict(X_grid), color='blue')

# Visualizing the polynomial regression
plt.plot(X_grid, pol_lin_regressor.predict(pol_regressor.fit_transform(X_grid)), color='green')

plt.show()

# Predicting a new result
lin_regressor.predict(6.5)
pol_lin_regressor.predict(pol_regressor.fit_transform(6.5))