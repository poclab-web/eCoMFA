import numpy as np
import cupy as cp
import scipy.linalg
import time

# 行列のサイズ
N = 1000

# NumPyでの計算
A_np = np.random.rand(N, N)
start_time = time.time()
eigvals_np, eigvecs_np = np.linalg.eig(A_np)
time_np = time.time() - start_time

# SciPyでの計算
start_time = time.time()
eigvals_scipy, eigvecs_scipy = scipy.linalg.eig(A_np)
time_scipy = time.time() - start_time

# CuPyでの計算
A_cp = cp.random.rand(N, N)
start_time = time.time()
eigvals_cp, eigvecs_cp = cp.linalg.eig(A_cp)
time_cp = time.time() - start_time

# 結果の表示
print(f"NumPy: {time_np:.5f} seconds")
print(f"SciPy: {time_scipy:.5f} seconds")
print(f"CuPy: {time_cp:.5f} seconds")
