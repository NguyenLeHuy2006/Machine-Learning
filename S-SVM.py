from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from cvxopt import matrix, solvers

# ==========================================
# 1. Khai báo thư viện và tạo dữ liệu giả
# ==========================================
np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.7, 0], [0, .7]]
N = 20 # number of samples per class
X0 = np.random.multivariate_normal(means[0], cov, N) # each row is a data point
X1 = np.random.multivariate_normal(means[1], cov, N)
X = np.concatenate((X0, X1))
y = np.concatenate((np.ones(N), -np.ones(N)))

# ==========================================
# 2. Giải bài toán bằng thư viện sklearn
# ==========================================
print("--- 1. Sklearn ---")
C = 100
clf = SVC(kernel='linear', C=C)
clf.fit(X, y)
w_sklearn = clf.coef_.reshape(-1, 1)
b_sklearn = clf.intercept_[0]
print('w_sklearn = \n', w_sklearn.T)
print('b_sklearn = ', b_sklearn)


# ==========================================
# 3. Tìm nghiệm bằng cách giải bài toán đối ngẫu
# ==========================================
print("\n--- 2. Dual Problem (CVXOPT) ---")
# build K
V = np.concatenate((X0, -X1), axis=0) # V[n, :] = y[n]*X[n]
K = matrix(V.dot(V.T))
p = matrix(-np.ones((2*N, 1)))

# build A, b, G, h
G = matrix(np.vstack((-np.eye(2*N), np.eye(2*N))))
h = matrix(np.vstack((np.zeros((2*N, 1)), C*np.ones((2*N, 1)))))
A = matrix(y.reshape((-1, 2*N)))
b = matrix(np.zeros((1, 1)))

solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)

l = np.array(sol['x']).reshape(2*N) # lambda vector

# support set
S = np.where(l > 1e-5)[0]
S2 = np.where(l < .999*C)[0]
# margin set
M = [val for val in S if val in S2] # intersection of two lists

VS = V[S]       # shape (NS, d)
lS = l[S]       # shape (NS,)
w_dual = lS.dot(VS) # shape (d,)
yM = y[M]       # shape (NM,)
XM = X[M]       # shape (NM, d)
b_dual = np.mean(yM - XM.dot(w_dual)) # shape (1,)
print('w_dual = ', w_dual)
print('b_dual = ', b_dual)


# ==========================================
# 4. Tìm nghiệm bằng giải bài toán tối ưu không ràng buộc
# ==========================================
print("\n--- 3. Hinge Loss & Gradient Descent ---")
lam = 1./C
def loss(X, y, w, b):
    """
    X.shape = (2N, d), y.shape = (2N,), w.shape = (d,), b is a scalar
    """
    z = X.dot(w) + b # shape (2N,)
    yz = y*z
    return (np.sum(np.maximum(0, 1 - yz)) + .5*lam*w.dot(w))/X.shape[0]

def grad(X, y, w, b):
    z = X.dot(w) + b # shape (2N,)
    yz = y*z
    active_set = np.where(yz <= 1)[0] # consider 1 - yz >= 0 only
    _yx = - X*y[:, np.newaxis]   # each row is y_n*x_n
    grad_w = (np.sum(_yx[active_set], axis=0) + lam*w)/X.shape[0]
    grad_b = (-np.sum(y[active_set]))/X.shape[0]
    return (grad_w, grad_b)

def num_grad(X, y, w, b):
    eps = 1e-10
    gw = np.zeros_like(w)
    gb = 0
    for i in range(len(w)):
        wp = w.copy()
        wm = w.copy()
        wp[i] += eps
        wm[i] -= eps
        gw[i] = (loss(X, y, wp, b) - loss(X, y, wm, b))/(2*eps)
    gb = (loss(X, y, w, b + eps) - loss(X, y, w, b - eps))/(2*eps)
    return (gw, gb)

# Kiểm tra đạo hàm
w = .1*np.random.randn(X.shape[1])
b = np.random.randn()
(gw0, gb0) = grad(X, y, w, b)
(gw1, gb1) = num_grad(X, y, w, b)
print('grad_w difference = ', np.linalg.norm(gw0 - gw1))
print('grad_b difference = ', np.linalg.norm(gb0 - gb1))

def softmarginSVM_gd(X, y, w0, b0, eta):
    w = w0
    b = b0
    it = 0
    while it < 10000:
        it = it + 1
        (gw, gb) = grad(X, y, w, b)
        w -= eta*gw
        b -= eta*gb
        if (it % 1000) == 0:
            print('iter %d' %it + ' loss: %f' %loss(X, y, w, b))
    return (w, b)

w0 = .1*np.random.randn(X.shape[1])
b0 = .1*np.random.randn()
lr = 0.05
(w_hinge, b_hinge) = softmarginSVM_gd(X, y, w0, b0, lr)
print('w_hinge = ', w_hinge)
print('b_hinge = ', b_hinge)


# ==========================================
# 5. Hàm hỗ trợ vẽ biểu đồ để tạo ra Hình 27.4 và Hình 27.5
# ==========================================
def plot_svm(X0, X1, w, b, ax, title):
    # Vẽ các điểm dữ liệu
    ax.plot(X0[:, 0], X0[:, 1], 'bs', markersize=6, alpha=0.8)
    ax.plot(X1[:, 0], X1[:, 1], 'ro', markersize=6, alpha=0.8)

    # Tạo meshgrid để vẽ đường biên
    x1_min, x1_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
    x2_min, x2_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200), np.linspace(x2_min, x2_max, 200))
    
    # Tính giá trị để phân lớp
    Z = w[0] * xx1 + w[1] * xx2 + b
    
    # Vẽ margins và boundary
    ax.contour(xx1, xx2, Z, colors='k', levels=[-1, 0, 1], alpha=0.8, linestyles=['-', '-', '-'], linewidths=[1, 2, 1])
    
    # Đổ màu nền
    ax.contourf(xx1, xx2, Z, levels=[-100, 0, 100], colors=['#E0E0FF', '#FFE0E0'], alpha=0.5)
    
    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)

# --- VẼ HÌNH 27.4 ---
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
plot_svm(X0, X1, w_sklearn.flatten(), b_sklearn, axes1[0], "Solution found by sklearn")
plot_svm(X0, X1, w_dual.flatten(), b_dual, axes1[1], "Solution found by dual")
plot_svm(X0, X1, w_hinge.flatten(), b_hinge, axes1[2], "Solution found by hinge")
fig1.suptitle('Hình 27.4: Các đường phân chia tìm được bởi 3 cách khác nhau')
plt.tight_layout()

# --- VẼ HÌNH 27.5 ---
C_values = [0.1, 1, 10, 100]
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
for c_val, ax in zip(C_values, axes2.flatten()):
    clf_c = SVC(kernel='linear', C=c_val)
    clf_c.fit(X, y)
    w_c = clf_c.coef_.flatten()
    b_c = clf_c.intercept_[0]
    plot_svm(X0, X1, w_c, b_c, ax, f"C = {c_val}")
fig2.suptitle('Hình 27.5: Ảnh hưởng của C lên nghiệm của soft-margin SVM')
plt.tight_layout()

# Hiển thị tất cả biểu đồ
plt.show()