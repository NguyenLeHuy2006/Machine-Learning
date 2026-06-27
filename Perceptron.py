import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. HÀM DỰ ĐOÁN VÀ THUẬT TOÁN PERCEPTRON
# ==========================================
def predict(w, X):
    '''Dự đoán nhãn (1 hoặc -1) dựa vào dấu của phép nhân ma trận'''
    return np.sign(X.dot(w))

def perceptron(X, y, w_init):
    '''Thuật toán Perceptron Learning Algorithm'''
    # Lưu lại lịch sử các trọng số w để theo dõi quá trình
    w = [w_init] 
    while True:
        pred = predict(w[-1], X)
        
        # Tìm index của các điểm bị phân loại sai
        mis_idxs = np.where(np.equal(pred, y) == False)[0]
        num_mis = mis_idxs.shape[0]
        
        # Nếu không còn điểm nào sai -> Thuật toán hội tụ, thoát vòng lặp
        if num_mis == 0: 
            return w
            
        # Chọn ngẫu nhiên 1 điểm bị phân loại sai để cập nhật
        random_id = np.random.choice(mis_idxs, 1)[0]
        
        # Cập nhật w mới: w_mới = w_cũ + y_i * x_i
        w_new = w[-1] + y[random_id]*X[random_id]
        w.append(w_new)

# ==========================================
# 2. TẠO DỮ LIỆU MẪU (2 CLASS)
# ==========================================
np.random.seed(2) # Cố định seed để ra kết quả giống nhau
means = [[-1, 0], [1, 0]]
cov = [[.3, .2], [.2, .3]]
N = 10

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)

X = np.concatenate((X0, X1), axis = 0)
y = np.concatenate((np.ones(N), -1*np.ones(N)))

# Thêm một cột chứa toàn số 1 vào bên trái ma trận X (thủ thuật bias trick)
Xbar = np.concatenate((np.ones((2*N, 1)), X), axis = 1)
w_init = np.random.randn(Xbar.shape[1])

# ==========================================
# 3. CHẠY THỬ THUẬT TOÁN
# ==========================================
w_history = perceptron(Xbar, y, w_init)

print(f"Thuật toán PLA đã hội tụ sau {len(w_history) - 1} vòng lặp cập nhật.")
print("Vector trọng số w cuối cùng:", w_history[-1])