import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
#dodajemy jedynki na poczatku wierszy X
X_train_1 = np.column_stack((np.ones_like(x_train), x_train))

# theta_best - wektor rozwiazan
# rozwiązanie jawne θ = (X^T * X)^(−1) * X^T * y gdzie X ma dodane jedynki na poczatku wierszy
theta_best = np.linalg.inv(X_train_1.T.dot(X_train_1)).dot(X_train_1.T).dot(y_train)

# TODO: calculate error

MSE_error = 0


for i in range(np.size(y_test)):
    MSE_error += ((float(theta_best[0]) + float(theta_best[1]) * x_test[i]) - y_test[i])**2

MSE_error /= np.size(y_test)

print("MSE error regresja liniowa:", MSE_error)
# print("Theta regresja:", theta_best[0], theta_best[1])

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization
# mean - srednia, std- odchylenie standardowe
mean_x_train = np.mean(x_train)
mean_y_train = np.mean(y_train)
std_x_train = np.std(x_train)
std_y_train = np.std(y_train)

x_train_normalized = (x_train - mean_x_train) / std_x_train
y_train_normalized = (y_train - mean_y_train) / std_y_train

x_test_normalized = (x_test - mean_x_train) / std_x_train
y_test_normalized = (y_test - mean_y_train) / std_y_train
# TODO: calculate theta using Batch Gradient Descent

theta = np.random.rand(2)

learning_rate = 0.01

MSE_train = 0

for i in range(np.size(y_train)):
    MSE_train += ((theta[0] + theta[1] * x_train_normalized[i]) - y_train_normalized[i]) ** 2

MSE_train /= np.size(y_train)

MSE_previous = MSE_train + 1

while(MSE_previous > MSE_train):

    x_train_normalized_1 = np.column_stack((np.ones_like(x_train_normalized), x_train_normalized))

    gradient_MSE = (2/np.size(y_train)) * x_train_normalized_1.T.dot(x_train_normalized_1.dot(theta) - y_train_normalized)

    theta = theta - learning_rate*gradient_MSE

    MSE_previous = MSE_train

    MSE_train = 0

    for i in range(np.size(y_train)):
        MSE_train += ((float(theta[0]) + float(theta[1]) * x_train_normalized[i]) - y_train_normalized[i])**2

    MSE_train /= np.size(y_train)

# TODO: calculate error
MSE_gradient = 0
for i in range(np.size(y_test)):
    MSE_gradient += ((float(theta[0]) + float(theta[1]) * x_test_normalized[i]) - y_test_normalized[i])**2

MSE_gradient /= np.size(y_test)

print("MSE error gradient:", MSE_gradient)
# print("Theta gradient:", theta[0], theta[1])

# plot the regression line
x = np.linspace(min(x_test_normalized), max(x_test_normalized), 100)
y = float(theta[0]) + float(theta[1]) * x
plt.plot(x, y)
plt.scatter(x_test_normalized, y_test_normalized)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
