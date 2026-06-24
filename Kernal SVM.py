import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons

# ==========================================
# 1. Bài toán XOR
# ==========================================
print("--- 1. Giải bài toán XOR với các Kernel ---")

# XOR dataset and targets
X_xor = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y_xor = np.array([0, 0, 1, 1])

# Khởi tạo đồ thị
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
kernels = ('sigmoid', 'poly', 'rbf')

for i, kernel in enumerate(kernels):
    # fit the model
    clf = svm.SVC(kernel=kernel, gamma=4, coef0=0)
    clf.fit(X_xor, y_xor)
    
    ax = axes[i]
    # Tạo lưới điểm để vẽ đường mức
    xx, yy = np.meshgrid(np.linspace(-1, 2, 200), np.linspace(-1, 2, 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Vẽ các vùng phân lớp và đường biên (margin)
    ax.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#E0E0FF', '#FFE0E0'], alpha=0.5)
    ax.contour(xx, yy, Z, colors='k', levels=[-0.5, 0, 0.5], alpha=0.8, linestyles=['--', '-', '--'], linewidths=[1, 1.5, 1])
    
    # Vẽ các điểm dữ liệu XOR
    ax.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], c='b', marker='s', edgecolors='k', label='Class 0')
    ax.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], c='r', marker='o', edgecolors='k', label='Class 1')
    
    ax.set_title(f"({chr(97+i)}) {kernel} kernel.")
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("Hình 28.2: Sử dụng kernel SVM để giải quyết bài toán XOR", fontsize=14)
plt.tight_layout()
plt.show()

# ==========================================
# Dữ liệu gần linearly separable
# ==========================================

# Tạo dữ liệu giả lập có độ cong nhẹ, gần phân biệt tuyến tính
X, y = make_moons(n_samples=60, noise=0.15, random_state=1)

# Khởi tạo đồ thị
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
kernels = ('sigmoid', 'poly', 'rbf')

for i, kernel in enumerate(kernels):
    # Khởi tạo mô hình SVM với các kernel khác nhau
    # coef0=1 giúp kernel 'poly' và 'sigmoid' linh hoạt hơn
    # gamma='scale' là mặc định tốt cho hầu hết trường hợp
    clf = svm.SVC(kernel=kernel, C=1.0, gamma=1.0, coef0=1)
    clf.fit(X, y)
    
    ax = axes[i]
    
    # Tạo lưới điểm (meshgrid) để vẽ không gian quyết định
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    # Tính giá trị dự đoán cho từng điểm trên lưới
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Đổ màu nền cho 2 class
    ax.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#FFE0E0', '#E0E0FF'], alpha=0.5)
    
    # Vẽ đường phân lớp (0) và đường đồng mức (+- 0.5 theo sách)
    ax.contour(xx, yy, Z, colors='k', levels=[-0.5, 0, 0.5], 
               alpha=0.8, linestyles=['--', '-', '--'], linewidths=[1, 1.5, 1])
    
    # Vẽ các điểm dữ liệu
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='r', marker='o', edgecolors='k', label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='b', marker='s', edgecolors='k', label='Class 1')
    
    ax.set_title(f"({chr(97+i)}) {kernel} kernel.")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle("Hình 28.3: Sử dụng kernel SVM để giải quyết bài toán với dữ liệu gần phân biệt tuyến tính", fontsize=14)
plt.tight_layout()
plt.show()