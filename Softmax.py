import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Đang tải dữ liệu MNIST")
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)

# Chuẩn hóa dữ liệu ảnh từ [0, 255] về [0, 1] giúp mô hình hội tụ nhanh hơn
X = mnist.data / 255.0 
y = mnist.target

# Chia tập dữ liệu (10000 mẫu để test, 60000 mẫu để train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

print("Đang huấn luyện mô hình Softmax Regression")
# C = 1e5 (nghịch đảo của lambda), solver = 'lbfgs', multi_class = 'multinomial' bật chế độ Softmax
# Tăng max_iter lên 1000 để thuật toán lbfgs có đủ thời gian chạm tới đáy hàm mất mát
model = LogisticRegression(C=1e5, solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# Dự đoán và tính độ chính xác
y_pred = model.predict(X_test)
print(f"Accuracy: {100 * accuracy_score(y_test, y_pred):.2f} %")


import numpy as np
import matplotlib.pyplot as plt

# Cố định random seed để hình ra giống nhau mỗi lần chạy
np.random.seed(1) 


def softmax(Z):
    # Kỹ thuật trừ đi max(Z) để tránh tràn số (overflow) khi tính exp
    e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return e_Z / e_Z.sum(axis=1, keepdims=True)

def softmax_loss(X, y, W):
    A = softmax(X.dot(W))
    id0 = range(X.shape[0])
    return -np.mean(np.log(A[id0, y]))

def softmax_fit(X, y, W, batch_size=10, nepoches=100, lr=0.05):
    """Huấn luyện bằng Mini-batch Gradient Descent"""
    W_old = W.copy()
    loss_hist = [softmax_loss(X, y, W)]
    N = X.shape[0]
    nbatches = int(np.ceil(float(N)/batch_size))
    
    for epoch in range(nepoches):
        mix_ids = np.random.permutation(N)
        for i in range(nbatches):
            batch_ids = mix_ids[batch_size*i : min(batch_size*(i+1), N)]
            X_batch = X[batch_ids]
            y_batch = y[batch_ids]
            
            # Tính Gradient và cập nhật W
            A = softmax(X_batch.dot(W))
            A[range(X_batch.shape[0]), y_batch] -= 1
            W -= lr * X_batch.T.dot(A) / batch_size
            
        loss_hist.append(softmax_loss(X, y, W))
    return W, loss_hist

def predict(X, W):
    A = softmax(X.dot(W))
    return np.argmax(A, axis=1)


C, N = 5, 500 # 5 Lớp, mỗi lớp 500 điểm
means = [[2, 2], [8, 3], [3, 6], [14, 2], [12, 8]]
cov = [[1, 0], [0, 1]]

X0 = np.random.multivariate_normal(means[0], cov, N) # Xanh dương (Blue)
X1 = np.random.multivariate_normal(means[1], cov, N) # Xanh ngọc (Cyan)
X2 = np.random.multivariate_normal(means[2], cov, N) # Xanh lá (Green)
X3 = np.random.multivariate_normal(means[3], cov, N) # Vàng (Yellow)
X4 = np.random.multivariate_normal(means[4], cov, N) # Đỏ (Red)

X = np.concatenate((X0, X1, X2, X3, X4), axis=0)
Xbar = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1) # Bias trick

y = np.asarray([0]*N + [1]*N + [2]*N + [3]*N + [4]*N)

W_init = np.random.randn(Xbar.shape[1], C)
W, loss_hist = softmax_fit(Xbar, y, W_init, batch_size=10, nepoches=100, lr=0.05)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(loss_hist, linewidth=1.5)
ax1.set_xlabel('number of epoches')
ax1.set_ylabel('loss')
ax1.set_xlim([-5, 105])

xm = np.arange(-2, 18, 0.05)
ym = np.arange(-3, 12, 0.05)
xx, yy = np.meshgrid(xm, ym)

grid_points = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape[0])]
Z = predict(grid_points, W)
Z = Z.reshape(xx.shape)

from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#E6E6FA', '#D0FFFF', '#F0FFF0', '#FFFACD', '#FFE4E1'])
ax2.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

ax2.contour(xx, yy, Z, colors='k', linewidths=0.5)

ax2.plot(X0[:, 0], X0[:, 1], 'b^', markersize=4, alpha=0.8) # Xanh dương - Tam giác
ax2.plot(X1[:, 0], X1[:, 1], 'co', markersize=4, alpha=0.8) # Xanh ngọc - Tròn
ax2.plot(X2[:, 0], X2[:, 1], 'gs', markersize=4, alpha=0.8) # Xanh lá - Vuông
ax2.plot(X3[:, 0], X3[:, 1], 'y.', markersize=4, alpha=0.8) # Vàng - Chấm
ax2.plot(X4[:, 0], X4[:, 1], 'r*', markersize=4, alpha=0.8) # Đỏ - Ngôi sao

ax2.set_xlim([-1, 16])
ax2.set_ylim([-2, 11])
ax2.set_xticks([]) # Tắt nhãn trục x
ax2.set_yticks([]) # Tắt nhãn trục y

plt.tight_layout()
plt.show()