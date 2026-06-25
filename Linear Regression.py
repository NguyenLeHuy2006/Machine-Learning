import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model

# Chiều cao (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T

# Cân nặng (kg)
y = np.array([ 49, 50, 51,  54,  58,  59,  60,  62,  63,  64,  66,  67,  68])

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1) 

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)


w = np.dot(np.linalg.pinv(A), b)

w_0, w_1 = w[0], w[1]
print(f"Nghiệm tìm được bằng công thức: w_1 = {w_1:.8f}, w_0 = {w_0:.8f}")

y1 = w_1 * 155 + w_0
y2 = w_1 * 160 + w_0
print(f"Input 155cm, true output 52kg, predicted output {y1:.2f}kg")
print(f"Input 160cm, true output 56kg, predicted output {y2:.2f}kg\n")





regr = linear_model.LinearRegression()
regr.fit(X, y)

print(f"Nghiệm của scikit-learn: w_1 = {regr.coef_[0]:.8f}, w_0 = {regr.intercept_:.8f}")






plt.plot(X, y, 'ro')

x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1 * x0

plt.plot(x0, y0, 'b-')

plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.axis([140, 190, 45, 75])
plt.title('Linear Regression: Dự đoán cân nặng theo chiều cao')
plt.show()