import pycuda.autoinit
import time
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule


TILE_DIM = 16


mod = SourceModule(open("kernel.cu").read(), options=['-std=c++11'])
def dot_product(v1, v2):
    acc=0
    for v_1,v_2 in zip(v1,v2):
        acc+=v_1*v_2
    return acc
def get_columns(W, index):
    res=[]
    for row in W:
        res.append(row[index])
    return res
def matrix_multiplication(X, W):
    res=[]
    for row_x in X:
        tmp=[]
        for i in range(len(W[0])):
            tmp.append(dot_product(row_x,get_columns(W,i))) #Dot product between each row of the matrix X and each line of the matrix W
        res.append(tmp)

    return res
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


a=4096
b=4096
c=4096
A = np.random.randn(a, b).astype(np.float32)
B = np.random.randn(b, c).astype(np.float32)
start_gpu=time.time()
C = matmul_gpu(A, B)
end_gpu=time.time()

print("Matrice A:")
print(A)
print("Matrice B:")
print(B)
print("Matrice C (résultat):")
print(C)
print("Gpu Time :" ,end_gpu-start_gpu)

#start_cpu=time.time()
#C_cpu=matrix_multiplication(A,B)
#end_cpu=time.time()
start_np=time.time()
C_np=np.dot(A,B)
end_np=time.time()
print("C error (réponse):")
print(C-C_np)
print("Numpy Time :" ,end_np-start_np)
#print("CPU Time :" ,end_cpu-start_cpu)

