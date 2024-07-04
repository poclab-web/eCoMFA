import numpy as np
import scipy.linalg
from time import time
from joblib import Parallel, delayed

# 固有値と固有ベクトルの計算を行う関数
def compute_eigen(matrix):
    return scipy.linalg.eigh(matrix)

# 並列計算を行う関数
def parallel_compute_eigen(matrix, num_jobs):
    n = matrix.shape[0]
    results = Parallel(n_jobs=num_jobs)(delayed(scipy.linalg.eigh)(matrix) for _ in range(num_jobs))
    return results

# 行列のサイズ
N = 2000

# ランダムな行列を生成
A = np.random.rand(N, N)
A=A.T@A
# 通常の固有値と固有ベクトルの計算時間を測定
start_time = time()
eigenvalues, eigenvectors = compute_eigen(A)
end_time = time()
print(f"通常の固有値と固有ベクトルの計算時間: {end_time - start_time} 秒")

# 並列計算の固有値と固有ベクトルの計算時間を測定
num_jobs = 50  # 使用する並列ジョブの数
start_time = time()
parallel_results = parallel_compute_eigen(A, num_jobs)
end_time = time()
print(f"並列計算の固有値と固有ベクトルの計算時間: {end_time - start_time} 秒")
