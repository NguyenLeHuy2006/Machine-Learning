import numpy as np

# 1. Hàm huấn luyện (Training)
def multiclass_svm_GD(X, y, Winit, reg, lr=1e-1, batch_size=1000, num_iters=50, print_every=10):
    W = Winit
    loss_history = []
    
    for it in range(num_iters):
        # Trộn ngẫu nhiên dữ liệu
        mix_ids = np.random.permutation(X.shape[0])
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        
        for ib in range(n_batches):
            # Lấy 1 mini-batch
            ids = mix_ids[batch_size*ib : min(batch_size*(ib+1), X.shape[0])]
            X_batch = X[ids]
            y_batch = y[ids]
            
            # Tính loss và gradient (dùng hàm vectorized ở phần trước)
            lossib, dw = svm_loss_vectorized(W, X_batch, y_batch, reg)
            loss_history.append(lossib)
            
            # CẬP NHẬT TRỌNG SỐ
            W -= lr * dw 
            
        if it % print_every == 0 and it > 0:
            print('it %d/%d, loss = %f' % (it, num_iters, loss_history[it]))
            
    return W, loss_history

# 2. Hàm dự đoán và tính độ chính xác
def multisvm_predict(W, X):
    Z = X.dot(W)
    # Lấy class có điểm số cao nhất
    return np.argmax(Z, axis=1) 

def evaluate(W, X, y):
    y_pred = multisvm_predict(W, X)
    acc = 100 * np.mean(y_pred == y)
    return acc

# 3. Vòng lặp tinh chỉnh siêu tham số (Hyperparameter Tuning)
# Giả sử X_train, y_train, X_val, y_val, X_test, y_test đã được chuẩn bị từ trước
d, C = X_train.shape[1], 10
W_init = 0.00001 * np.random.randn(d, C)

lrs = [1e-9, 1e-8, 1e-7, 1e-6]
regs = [0.1, 0.01, 0.001, 0.0001]

best_W = None
best_acc = 0

print("Bắt đầu Grid Search:")
for lr in lrs:
    for reg in regs:
        # Huấn luyện thử với cặp (lr, reg)
        W, loss_history = multiclass_svm_GD(X_train, y_train, W_init, reg, 
                                            lr=lr, num_iters=100, print_every=1e20) # print_every=1e20 để ẩn log
        
        # Chấm điểm trên tập Validation
        acc = evaluate(W, X_val, y_val)
        print('lr = %e, reg = %e, loss = %f, validation acc = %.2f' % (lr, reg, loss_history[-1], acc))
        
        # Lưu lại mô hình xịn nhất
        if acc > best_acc:
            best_acc = acc
            best_W = W.copy() # Dùng copy để tránh lỗi tham chiếu

# 4. Đánh giá lần cuối trên Test Set
print("\n--- KẾT QUẢ CUỐI CÙNG ---")
acc_test = evaluate(best_W, X_test, y_test)
print('Accuracy on test data = %.2f %%' % acc_test)