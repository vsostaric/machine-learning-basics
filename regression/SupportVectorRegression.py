# SVR - support vector regression

# Data Preprocessing

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("../data/Position_Salaries.csv");
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predicting the test train result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))                                                  ))

# Visualizing the regression results
plt.plot(X, regressor.predict(X), color='blue')
plt.scatter(X, y, color='red')

# Creating the plot
plt.title('SVR')
plt.xlabel('Position')
plt.ylabel('Salary')

plt.show()