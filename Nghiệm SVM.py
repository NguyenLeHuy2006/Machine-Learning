import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.svm import SVC

# ==========================================
# 1. TẠO DỮ LIỆU MẪU (2 CLASS LINEARLY SEPARABLE)
# ==========================================
np.random.seed(22)
means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10

X0 = np.random.multivariate_normal(means[0], cov, N) # Lớp 1 (Màu xanh)
X1 = np.random.multivariate_normal(means[1], cov, N) # Lớp -1 (Màu đỏ)
X = np.concatenate((X0, X1), axis = 0)
y = np.concatenate((np.ones(N), -np.ones(N)), axis = 0) # Nhãn (1 và -1)

# ==========================================
# 2. CÁCH 1: GIẢI BẰNG TOÁN HỌC (THƯ VIỆN CVXOPT)
# ==========================================
print("--- 1. NGHIỆM THEO CÔNG THỨC TOÁN (CVXOPT) ---")
# Tính toán ma trận V (gộp X0 và -X1)
V = np.concatenate((X0, -X1), axis = 0)

# Thiết lập các tham số cho bài toán Quadratic Programming
Q = matrix(V.dot(V.T))
p = matrix(-np.ones((2*N, 1)))
G = matrix(-np.eye(2*N))
h = matrix(np.zeros((2*N, 1)))
A = matrix(y.reshape(1, -1))
b = matrix(np.zeros((1, 1)))

solvers.options['show_progress'] = False
sol = solvers.qp(Q, p, G, h, A, b)
l = np.array(sol['x']) # Nghiệm lambda

# Tính toán trọng số w và bias b
S = np.where(l > 1e-8)[0] # Support set (chỉ lấy các điểm có lambda > 0)
w_math = V.T.dot(l)
b_math = np.mean(y[S].reshape(-1, 1) - X[S, :].dot(w_math))

print('Số lượng Support Vectors = ', S.size)
print('w = ', w_math.T)
print('b = ', b_math)


# ==========================================
# 3. CÁCH 2: GIẢI BẰNG SCIKIT-LEARN
# ==========================================
print("\n--- 2. NGHIỆM THEO THƯ VIỆN SCIKIT-LEARN ---")
# C = 1e5 là một số rất lớn (Hard Margin SVM)
model = SVC(kernel = 'linear', C = 1e5) 
model.fit(X, y)

w_sklearn = model.coef_
b_sklearn = model.intercept_

print('w = ', w_sklearn)
print('b = ', b_sklearn)


# ==========================================
# 4. VẼ ĐỒ THỊ MINH HỌA SVM 
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6))

# Vẽ vùng nền (lãnh thổ)
x1_min, x1_max = 0.5, 5.5
x2_min, x2_max = 0.5, 3.5
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                       np.arange(x2_min, x2_max, 0.02))
Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)
ax.contourf(xx1, xx2, Z, cmap=plt.cm.bwr, alpha=0.2)

# Vẽ các điểm dữ liệu
ax.plot(X0[:, 0], X0[:, 1], 'bs', markersize=6)
ax.plot(X1[:, 0], X1[:, 1], 'ro', markersize=6)

# Khoanh tròn các Support Vectors
ax.scatter(X[S, 0], X[S, 1], s=150, facecolors='none', edgecolors='k', linewidths=1.5)

# Vẽ đường ranh giới (Decision Boundary) và lề (Margins)
w = w_sklearn[0]
b = b_sklearn[0]
x1_vals = np.array([x1_min, x1_max])
x2_vals = -(w[0] * x1_vals + b) / w[1] # Đường ranh giới chính (w.x + b = 0)
x2_margin_pos = -(w[0] * x1_vals + b - 1) / w[1] # Đường lề dương (w.x + b = 1)
x2_margin_neg = -(w[0] * x1_vals + b + 1) / w[1] # Đường lề âm (w.x + b = -1)

ax.plot(x1_vals, x2_vals, 'k-', linewidth=2)
ax.plot(x1_vals, x2_margin_pos, 'k-', linewidth=1)
ax.plot(x1_vals, x2_margin_neg, 'k-', linewidth=1)

ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min, x2_max)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
plt.title('Minh họa Support Vector Machine (SVM)')
plt.show()