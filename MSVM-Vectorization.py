import numpy as np
import time

def svm_loss_vectorized(W, X, y, reg):
    """
    Tính loss và gradient của hàm mất mát. Cách tối ưu (Vectorized way)
    """
    d, C = W.shape
    N = X.shape[0]
    loss = 0
    dw = np.zeros_like(W)
    
    # Bước 1: Tính ma trận Score Z (N, C)
    Z = X.dot(W) 
    
    # Bước 2: Lấy điểm của các lớp đúng ra thành một vector cột
    id0 = np.arange(Z.shape[0])
    correct_class_score = Z[id0, y].reshape(N, 1) 
    
    # Tính margin cho toàn bộ ma trận (áp dụng công thức Hinge Loss)
    margins = np.maximum(0, Z - correct_class_score + 1)
    
    # Ép margin của lớp đúng về 0 (vì ta không tính loss của nó với chính nó)
    margins[id0, y] = 0
    
    # Tổng hợp Data loss và Regularization loss
    loss = np.sum(margins) / N
    loss += 0.5 * reg * np.sum(W * W)
    
    # Bước 3 & 4: Tính ma trận đạo hàm F
    F = (margins > 0).astype(int) # Chuyển các ô có vi phạm thành 1, ngược lại là 0
    
    # Cập nhật hệ số đạo hàm cho các ô đúng (bằng âm tổng số lần vi phạm trong cùng cột/dữ liệu)
    F[np.arange(F.shape[0]), y] = np.sum(-F, axis=1) 
    
    # Tính ma trận đạo hàm cuối cùng dw
    dw = X.T.dot(F) / N + reg * W
    
    return loss, dw

# ==========================================
# Test thử với dữ liệu ngẫu nhiên (Sanity check)
# Giả lập dữ liệu lớn để test 
d, C = 3073, 10
N_train = 49000
W_rand = np.random.randn(d, C)
X_train = np.random.randn(N_train, d)
y_train = np.random.randint(0, C, N_train)
reg = 0.1

# Bắt đầu đo thời gian
t1 = time.time()
l1, dw1 = svm_loss_naive(W_rand, X_train, y_train, reg)
t2 = time.time()
l2, dw2 = svm_loss_vectorized(W_rand, X_train, y_train, reg)
t3 = time.time()

print('Naive       -- run time:', t2 - t1, '(s)')
print('Vectorized  -- run time:', t3 - t2, '(s)')
# Tính độ lệch giữa 2 cách tính (càng nhỏ càng tốt, lý tưởng là 0)
print('loss difference:', np.linalg.norm(l1 - l2))
print('gradient difference:', np.linalg.norm(dw1 - dw2))