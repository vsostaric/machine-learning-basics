import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("../data/Position_Salaries.csv");
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

# Creating the Model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,
                                  random_state=0)

# Fitting the Model
regressor.fit(X, y)

# Predicting the test train result
y_pred = regressor.predict(6.5)

# Griding for higher resolution (replace X in plot with X_grid)
resolution = 1000
X_grid = np.arange(min(X), max(X), (max(X)-min(X))/resolution)
X_grid = X_grid.reshape(len(X_grid), 1)

# Creating the plot
plt.title('Position Vs Salary')
plt.xlabel('Position')
plt.ylabel('Salary')

# Visualizing the regression results
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')

# Show Plot
plt.show()