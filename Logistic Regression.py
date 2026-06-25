import numpy as np
import matplotlib.pyplot as plt


def sigmoid(S):
    return 1 / (1 + np.exp(-S))

def prob(w, X):
    """Tính xác suất dự đoán dựa trên ma trận X và trọng số w"""
    return sigmoid(X.dot(w))

def loss(w, X, y, lam):
    """Tính Cross-Entropy Loss kèm theo L2 Regularization (weight decay)"""
    z = prob(w, X)
    z = np.clip(z, 1e-15, 1 - 1e-15) 
    return -np.mean(y*np.log(z) + (1-y)*np.log(1-z)) + 0.5*lam/X.shape[0]*np.sum(w*w)

def logistic_regression(w_init, X, y, lam = 0.001, lr = 0.1, nepoches = 2000):
    """Học trọng số w bằng phương pháp Stochastic Gradient Descent (SGD)"""
    N, d = X.shape[0], X.shape[1]
    w = w_old = w_init
    loss_hist = [loss(w_init, X, y, lam)]
    ep = 0
    while ep < nepoches:
        ep += 1
        mix_ids = np.random.permutation(N) 
        for i in mix_ids:
            xi = X[i]
            yi = y[i]
            zi = sigmoid(xi.dot(w))
            w = w - lr*((zi - yi)*xi + lam*w)
        loss_hist.append(loss(w, X, y, lam))
        # Dừng sớm nếu thuật toán đã hội tụ
        if np.linalg.norm(w - w_old)/d < 1e-6:
            break
        w_old = w
    return w, loss_hist


np.random.seed(2)

X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
               2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).T
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

N = X.shape[0]
Xbar = np.concatenate((X, np.ones((N, 1))), axis = 1)
w_init = np.random.randn(Xbar.shape[1])
lam = 0.0001


w, loss_hist = logistic_regression(w_init, Xbar, y, lam, lr = 0.05, nepoches = 500)

print("Solution of Logistic Regression:", w)
print("Final loss:", loss(w, Xbar, y, lam))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

X0 = X[y == 0]
X1 = X[y == 1]
ax1.plot(X0, np.zeros_like(X0), 'ro', markeredgecolor='k', markersize=8)
ax1.plot(X1, np.ones_like(X1), 'bs', markeredgecolor='k', markersize=8)

xx = np.linspace(0, 6, 1000).reshape(-1, 1)
xx_bar = np.concatenate((xx, np.ones((xx.shape[0], 1))), axis=1)
yy = prob(w, xx_bar)
ax1.plot(xx, yy, 'g-', linewidth=2)


threshold_x = -w[1]/w[0]
ax1.plot(threshold_x, 0.5, 'y^', markeredgecolor='k', markersize=8)

ax1.set_xlabel('studying hours')
ax1.set_ylabel('predicted probability of pass')
ax1.set_xlim(0, 6)
ax1.set_ylim(-0.5, 1.5)

ax2.plot(loss_hist, 'C0-')
ax2.set_xlabel('number of iterations')
ax2.set_ylabel('loss function')

plt.tight_layout()
plt.show()