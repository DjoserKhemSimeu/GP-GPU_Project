import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule


TILE_DIM = 16


mod = SourceModule(open("kernel.cu").read(), options=['-std=c++11'])


def matmul_gpu(A, B):
    ARows, ACols = A.shape
    BRows, BCols = B.shape
    CRows, CCols = ARows, BCols

    
    A_gpu = drv.mem_alloc(A.nbytes)
    B_gpu = drv.mem_alloc(B.nbytes)
    C_gpu = drv.mem_alloc(CRows * CCols * A.dtype.itemsize)

    
    drv.memcpy_htod(A_gpu, A)
    drv.memcpy_htod(B_gpu, B)

    
    matmul = mod.get_function("MatMul")

    
    block_dim = (TILE_DIM, TILE_DIM, 1)
    grid_dim = (int(np.ceil(CCols / TILE_DIM)), int(np.ceil(CRows / TILE_DIM)), 1)

    
    matmul(A_gpu, B_gpu, C_gpu, np.int32(ARows), np.int32(ACols), np.int32(BRows),
           np.int32(BCols), np.int32(CRows), np.int32(CCols),
           block=block_dim, grid=grid_dim)

    
    C = np.empty((CRows, CCols), dtype=A.dtype)
    drv.memcpy_dtoh(C, C_gpu)

    return C


a=pow(2,3)
b=pow(2,2)
c=pow(2,3)
A = np.random.randn(a, b).astype(np.float32)
B = np.random.randn(b, c).astype(np.float32)
C = matmul_gpu(A, B)

print("Matrice A:")
print(A)
print("Matrice B:")
print(B)
print("Matrice C (résultat):")
print(C)
print("C error (réponse):")
print(C-np.dot(A,B))

