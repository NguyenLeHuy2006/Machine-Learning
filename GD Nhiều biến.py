import numpy as np
from sklearn.linear_model import LinearRegression


np.random.seed(0) 
X = np.random.rand(1000)
y = 4 + 3 * X + .5 * np.random.randn(1000) 

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)


model = LinearRegression()
model.fit(X, y)
w_sklearn, b_sklearn = model.coef_[0][0], model.intercept_[0]
print("--- KẾT QUẢ TỪ SCIKIT-LEARN ---")
print(f"Solution: w_0 (bias) = {b_sklearn:.6f}, w_1 = {w_sklearn:.6f}\n")



one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)

def grad(w):
    N = Xbar.shape[0]
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost(w):
    N = Xbar.shape[0]
    return .5/N * np.linalg.norm(y - Xbar.dot(w))**2

def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(100):
        w_new = w[-1] - eta*grad(w[-1])
        
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break
        w.append(w_new)
    return (w, it)

w_init = np.array([[2.0], [1.0]])
(w_gd, it_gd) = myGD(w_init, grad, 1)

print("--- KẾT QUẢ TỪ GRADIENT DESCENT (eta = 1) ---")
print(f"Solution: w_0 (bias) = {w_gd[-1][0,0]:.6f}, w_1 = {w_gd[-1][1,0]:.6f}")
print(f"Chạy mất {it_gd+1} vòng lặp.")