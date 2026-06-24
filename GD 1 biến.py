import numpy as np
import matplotlib.pyplot as plt

def grad(x):
    return 2*x + 5*np.cos(x)

def cost(x):
    return x**2 + 5*np.sin(x)

def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

x0 = -5
eta = 0.1
(x_history, total_iters) = myGD1(x0, eta)

(x1, it1) = myGD1(-5, .1)
(x2, it2) = myGD1(5, .1)

print('Solution x1 = %f, cost = %f, after %d iterations' % (x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, after %d iterations' % (x2[-1], cost(x2[-1]), it2))


iters_to_plot = [0, 1, 2, 3, 4, 5, 7, 11]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

x_val = np.linspace(-6, 6, 100)
y_val = cost(x_val)

for i, it_val in enumerate(iters_to_plot):
    row = i // 4 
    col = i % 4  
    ax = axes[row, col]
    
    if it_val < len(x_history):
        xi = x_history[it_val]
        yi = cost(xi)
        grad_i = grad(xi)
        ax.plot(x_val, y_val, 'b-')
        ax.plot(xi, yi, 'ro', markersize=8, markeredgecolor='k')
        ax.set_xlabel(f'iter {it_val}/{total_iters}, grad = {grad_i:.3f}')
        
    ax.set_xlim([-6, 6])
    ax.set_ylim([-5, 35]) 

plt.tight_layout()
plt.show()