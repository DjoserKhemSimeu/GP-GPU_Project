import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# Kernel code (MatMul and add_bias) would be the same as provided earlier
mod = SourceModule(open("../kernel.cu").read(), options=['-std=c++11'])

# Define constants
TILE_DIM = 16
M = 4  # Number of rows in A and C
N = 3  # Number of columns in A and B
K = 2  # Number of columns in A and rows in B

# Initialize test matrices and bias
A = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)  # MxK matrix
B = np.array([[1, 0, -1], [1, 1, 1]], dtype=np.float32)  # KxN matrix
bias = np.array([0.5, -0.5, 1.0], dtype=np.float32)  # Bias vector (size N)

# Expected results
expected_result = np.array([[3, 2, 1], [7, 8, 9], [11, 14, 17], [15, 20, 25]], dtype=np.float32)  # Expected result for MatMul
expected_add_bias_result = np.array([[3.5, 1.5, 2.0], [7.5, 7.5, 10.0], [11.5, 13.5, 18.0], [15.5, 19.5, 26.0]], dtype=np.float32)

# Allocate memory for matrices and result arrays on the GPU
A_gpu = drv.mem_alloc(A.nbytes)
B_gpu = drv.mem_alloc(B.nbytes)
C_gpu = drv.mem_alloc(A.shape[0] * B.shape[1] * np.float32().itemsize)
bias_gpu = drv.mem_alloc(bias.nbytes)
C_add_bias_gpu = drv.mem_alloc(A.shape[0] * N * np.float32().itemsize)

# Copy matrices to the GPU
drv.memcpy_htod(A_gpu, A)
drv.memcpy_htod(B_gpu, B)
drv.memcpy_htod(bias_gpu, bias)

# Load the kernel code to the GPU
# (The kernel code should already be loaded above with `SourceModule`)

# Get kernel functions
matmul_kernel = mod.get_function('MatMul')
add_bias_kernel = mod.get_function('add_bias')

# Set the kernel execution parameters
block_size = (TILE_DIM, TILE_DIM, 1)

# Grid size for matrix multiplication
grid_size_matmul = ((N + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM)
# Grid size for add_bias
grid_size_add_bias = ((N + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM)

# Run matrix multiplication kernel
matmul_kernel(A_gpu, B_gpu, C_gpu, np.int32(M), np.int32(K), np.int32(N),
              block=block_size, grid=grid_size_matmul)

# Copy the result back to host
C = np.empty((M, N), dtype=np.float32)
drv.memcpy_dtoh(C, C_gpu)

# Compare results for matrix multiplication
print("Matrix Multiplication Result:")
print(C)
print("Expected Matrix Multiplication Result:")
print(np.dot(A,B))

# Run add_bias kernel
add_bias_kernel(C_gpu, bias_gpu, C_add_bias_gpu, np.int32(M), np.int32(N),
                 block=block_size, grid=grid_size_add_bias)

# Copy the result back to host
C_add_bias = np.empty((M, N), dtype=np.float32)
drv.memcpy_dtoh(C_add_bias, C_add_bias_gpu)

# Compare results for add_bias
print("\nAdd Bias Result:")
print(C_add_bias)
print("Expected Add Bias Result:")
print(np.dot(A,B)+bias)
