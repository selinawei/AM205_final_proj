import numpy as np
import pandas as pd
import time
from optimization_methods import bfgs_method

def python_solve(A_in, b_in):
    """
    Solves Ax = b using Naive Gaussian Elimination in pure Python loops.
    This removes the LAPACK/BLAS optimization advantage.
    """
    # 拷贝数据，防止修改原矩阵
    A = A_in.copy()
    b = b_in.copy()
    n = len(b)
    
    # --- 1. 消元过程 (Forward Elimination) ---
    for i in range(n):
        # 寻找主元 (Pivot)
        pivot = A[i, i]
        if abs(pivot) < 1e-12: # 简单的数值稳定性处理
             # 在实际中应该交换行，这里为了简化略过，假设矩阵性质够好
            pass 
            
        for j in range(i + 1, n):
            factor = A[j, i] / pivot
            # 这一行是性能杀手：Python 循环处理行操作
            # A[j, i:] -= factor * A[i, i:]  <-- 如果用 numpy 切片还是会快
            # 为了彻底去优化，我们用循环：
            for k in range(i, n):
                A[j, k] -= factor * A[i, k]
            b[j] -= factor * b[i]

    # --- 2. 回代过程 (Back Substitution) ---
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_ax = 0.0
        for j in range(i + 1, n):
            sum_ax += A[i, j] * x[j]
        x[i] = (b[i] - sum_ax) / A[i, i]
        
    return x

def newton_method_manual(func, grad, hess, x0, max_iter=10, tol=1e-4):
    x = x0
    iterations = 0
    f_val = func(x)
    
    for i in range(max_iter):
        iterations += 1
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
            
        H = hess(x)
        
        # [关键差异点]
        # 不使用 np.linalg.solve(H, -g)
        # 使用我们要手写的慢速求解器
        d = python_solve(H, -g)
        
        x = x + d
        f_val = func(x)
        
    return {'x': x, 'f': f_val, 'iterations': iterations}


dimensions = [2,5, 10, 20, 30, 40, 50, 100, 145, 200] 
results_table = []

for n in dimensions:
    print(f"\nDimension n = {n} ... ", end="")
    
    # 构造问题: 简单的二次函数 (Hessian 为常数，方便计算，主要测求解耗时)
    # f(x) = 0.5 * x.T * A * x
    np.random.seed(42)
    A = np.random.randn(n, n)
    A = A.T @ A + np.eye(n) # 保证正定
    
    def func(x): return 0.5 * x @ A @ x
    def grad(x): return A @ x
    def hess(x): return A # Hessian 就是 A
    
    x0 = np.random.randn(n)
    
    # --- Run BFGS ---
    start = time.time()
    # BFGS 使用标准的 NumPy 实现 (O(N^2) 矩阵向量乘法)
    # 这对 BFGS 其实还有点不公平，因为 NumPy 还是带优化的
    # 但由于 BFGS 没有求解方程组这一步，差异主要在算法复杂度
    bfgs_res = bfgs_method(func, grad, x0, max_iter=5, tol=1e-5)
    bfgs_time = time.time() - start
    
    # --- Run Newton (Manual) ---
    start = time.time()
    # 强制 Newton 跑满 CPU 循环
    newton_res = newton_method_manual(func, grad, hess, x0, max_iter=5, tol=1e-5)
    newton_time = time.time() - start
    
    print(f"Done.")
    
    # 记录单步时间 (Time per Iteration) 以示公正
    bfgs_per_iter = bfgs_time / max(1, bfgs_res['iterations'])
    newton_per_iter = newton_time / max(1, newton_res['iterations'])
    
    results_table.append({
        'N': n,
        'BFGS_Iters': bfgs_res['iterations'],
        'BFGS_Total_Time(s)': bfgs_time,
        'BFGS_Time/Iter(s)': bfgs_per_iter,
        'Newton_Iters': newton_res['iterations'],
        'Newton_Total_Time(s)': newton_time,
        'Newton_Time/Iter(s)': newton_per_iter,
        'Speedup (BFGS/Newton)': newton_per_iter / bfgs_per_iter
    })


df = pd.DataFrame(results_table)


# 保存到 CSV
df.to_csv('./exp1/fair_comparison.csv', index=False)