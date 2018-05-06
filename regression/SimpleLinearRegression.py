import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("../data/Salary_Data.csv")

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_trained = regressor.predict(X_train)

plt.scatter(X_train, y_train, color= 'red')
plt.plot(X_train, y_trained, color='blue')
plt.title('Salary Vs Experience (Train data)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

plt.show()

plt.scatter(X_test, y_test, color= 'red')
plt.plot(X_train, y_trained, color='blue')
plt.title('Salary Vs Experience (Test data)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

plt.show()