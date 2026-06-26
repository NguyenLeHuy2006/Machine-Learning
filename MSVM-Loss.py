import numpy as np

def svm_loss_naive(W, X, y, reg):
    """
    Tính loss và gradient (đạo hàm) của hàm mất mát tại W. Cách đơn giản (Naive way)
    W: ma trận trọng số, kích thước (d, C)
    X: dữ liệu training, kích thước (N, d)
    y: nhãn training, kích thước (N,)
    reg: tham số regularization
    """
    # Lấy kích thước dữ liệu (Sửa lại lỗi cú pháp của sách gốc)
    d, C = W.shape 
    N = X.shape[0] 
    
    loss = 0
    dw = np.zeros_like(W) # Ma trận chứa đạo hàm, cùng kích thước với W
    
    # Duyệt qua từng điểm dữ liệu
    for n in range(N): 
        xn = X[n]
        score = xn.dot(W) # Tính điểm cho tất cả các lớp của điểm dữ liệu xn
        
        # Duyệt qua từng class để tính Hinge Loss
        for j in range(C):
            if j == y[n]: # Bỏ qua nếu đây là lớp đúng
                continue
                
            # Tính khoảng cách vi phạm
            margin = 1 - score[y[n]] + score[j]
            
            # Nếu có vi phạm (margin > 0)
            if margin > 0:
                loss += margin
                # Cập nhật đạo hàm theo công thức 29.7 và 29.8
                dw[:, j] += xn     # Tăng trọng số của lớp đoán sai
                dw[:, y[n]] -= xn  # Giảm trọng số của lớp đoán đúng

    # Chia trung bình cho số lượng dữ liệu
    loss /= N
    dw /= N
    
    # Cộng thêm thành phần Regularization 
    loss += 0.5 * reg * np.sum(W * W)
    dw += reg * W
    
    return loss, dw

# ==========================================
# Test thử với dữ liệu ngẫu nhiên (Sanity check)
# ==========================================

# random, small data
d, C, N = 100, 3, 300
reg = 0.1
W_rand = np.random.randn(d, C)
X_rand = np.random.randn(N, d)
y_rand = np.random.randint(0, C, N)

# In kết quả
print('Loss with reg = 0  : ', svm_loss_naive(W_rand, X_rand, y_rand, 0)[0])
print('Loss with reg = 0.1: ', svm_loss_naive(W_rand, X_rand, y_rand, .1)[0])