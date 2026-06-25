import numpy as np
import time

# ==========================================
# 1. TẠO DỮ LIỆU GIẢ LẬP (Dummy Data)
# Giả lập dữ liệu đã qua bước Feature Engineering (có Bias trick)
# ==========================================
print("--- 1. CHUẨN BỊ DỮ LIỆU ---")
N_train, N_val, N_test = 4900, 1000, 1000 # Số lượng ảnh (thu nhỏ lại để chạy nhanh hơn)
d = 3073 # Kích thước vector ảnh (đã thêm 1 cột bias)
C = 10   # Số lượng class (10 loại)

# Tạo dữ liệu ngẫu nhiên
np.random.seed(42) # Cố định seed để ra kết quả giống nhau mỗi lần chạy
X_train = np.random.randn(N_train, d)
y_train = np.random.randint(0, C, N_train)
X_val = np.random.randn(N_val, d)
y_val = np.random.randint(0, C, N_val)
X_test = np.random.randn(N_test, d)
y_test = np.random.randint(0, C, N_test)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}\n")


# ==========================================
# 2. CÁC HÀM TÍNH LOSS VÀ GRADIENT
# ==========================================
def svm_loss_naive(W, X, y, reg):
    d, C = W.shape 
    N = X.shape[0] 
    loss = 0
    dw = np.zeros_like(W) 
    
    for n in range(N): 
        xn = X[n]
        score = xn.dot(W) 
        for j in range(C):
            if j == y[n]: 
                continue
            margin = 1 - score[y[n]] + score[j]
            if margin > 0:
                loss += margin
                dw[:, j] += xn     
                dw[:, y[n]] -= xn  
                
    loss /= N
    dw /= N
    loss += 0.5 * reg * np.sum(W * W)
    dw += reg * W
    return loss, dw

def svm_loss_vectorized(W, X, y, reg):
    d, C = W.shape
    N = X.shape[0]
    
    Z = X.dot(W) 
    id0 = np.arange(Z.shape[0])
    correct_class_score = Z[id0, y].reshape(N, 1) 
    
    margins = np.maximum(0, Z - correct_class_score + 1)
    margins[id0, y] = 0
    
    loss = np.sum(margins) / N
    loss += 0.5 * reg * np.sum(W * W)
    
    F = (margins > 0).astype(int) 
    F[np.arange(F.shape[0]), y] = np.sum(-F, axis=1) 
    
    dw = X.T.dot(F) / N + reg * W
    return loss, dw


# ==========================================
# 3. TEST TỐC ĐỘ (Sanity Check)
# ==========================================
print("--- 2. SO SÁNH TỐC ĐỘ NAIVE VS VECTORIZED ---")
W_rand = np.random.randn(d, C) * 0.0001
reg = 0.1

t1 = time.time()
l1, dw1 = svm_loss_naive(W_rand, X_train, y_train, reg)
t2 = time.time()
l2, dw2 = svm_loss_vectorized(W_rand, X_train, y_train, reg)
t3 = time.time()

print(f"Naive time      : {t2 - t1:.4f} (s)")
print(f"Vectorized time : {t3 - t2:.4f} (s)")
print(f"Difference      : {np.linalg.norm(l1 - l2):.10f}\n")


# ==========================================
# 4. HÀM HUẤN LUYỆN VÀ ĐÁNH GIÁ
# ==========================================
def multiclass_svm_GD(X, y, Winit, reg, lr=1e-1, batch_size=200, num_iters=50, print_every=10):
    W = Winit.copy()
    loss_history = []
    
    for it in range(num_iters):
        mix_ids = np.random.permutation(X.shape[0])
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        
        for ib in range(n_batches):
            ids = mix_ids[batch_size*ib : min(batch_size*(ib+1), X.shape[0])]
            X_batch = X[ids]
            y_batch = y[ids]
            
            lossib, dw = svm_loss_vectorized(W, X_batch, y_batch, reg)
            loss_history.append(lossib)
            W -= lr * dw 
            
        if it % print_every == 0 and it > 0:
            print(f'Iteration {it}/{num_iters}, loss = {loss_history[it]:.4f}')
            
    return W, loss_history

def multisvm_predict(W, X):
    Z = X.dot(W)
    return np.argmax(Z, axis=1) 

def evaluate(W, X, y):
    y_pred = multisvm_predict(W, X)
    return 100 * np.mean(y_pred == y)


# ==========================================
# 5. THỰC THI (GRID SEARCH)
# ==========================================
print("--- 3. HUẤN LUYỆN VÀ TÌM THAM SỐ TỐT NHẤT ---")
W_init = 0.00001 * np.random.randn(d, C)

lrs = [1e-3, 1e-4] # Tốc độ học (Giảm xuống để phù hợp với dữ liệu giả)
regs = [0.1, 0.01]   # Tham số phạt

best_W = None
best_acc = -1

for lr in lrs:
    for reg in regs:
        W, loss_history = multiclass_svm_GD(X_train, y_train, W_init, reg, 
                                            lr=lr, num_iters=10, print_every=100) # num_iters nhỏ để chạy lướt qua
        
        acc = evaluate(W, X_val, y_val)
        print(f'lr = {lr}, reg = {reg} => Validation acc = {acc:.2f}%')
        
        if acc > best_acc:
            best_acc = acc
            best_W = W.copy()

print("\n--- 4. KẾT QUẢ CUỐI CÙNG TRÊN TEST SET ---")
acc_test = evaluate(best_W, X_test, y_test)
print(f'Accuracy on test data = {acc_test:.2f}%')
# Lưu ý: Vì đây là dữ liệu ngẫu nhiên hoàn toàn (không có quy luật), 
# nên độ chính xác sẽ luôn loanh quanh ở mức 10% (1/10 class).

import matplotlib.pyplot as plt

# Lấy thử một quá trình training để vẽ đồ thị
# Ta gọi lại hàm train với số vòng lặp nhiều hơn một chút (ví dụ 1000 vòng) để thấy rõ độ dốc
print("\n--- VẼ ĐỒ THỊ LOSS (HÌNH 29.6) ---")
W_demo, loss_history_demo = multiclass_svm_GD(X_train, y_train, W_init, reg=0.1, lr=1e-3, num_iters=1000, print_every=200)

# Tiến hành vẽ đồ thị
plt.plot(loss_history_demo)
plt.xlabel('number of iterations') # Trục ngang là số vòng lặp
plt.ylabel('loss function')        # Trục dọc là giá trị hàm mất mát
plt.title('Lịch sử loss qua các vòng lặp (Hình 29.6)')
plt.show() # Lệnh này sẽ mở ra một cửa sổ popup hiển thị đồ thị