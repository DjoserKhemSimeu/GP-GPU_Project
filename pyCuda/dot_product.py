import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

def dot_product(v1, v2):
    acc=0
    for v_1,v_2 in zip(v1,v2):
        acc+=v_1*v_2
    return acc
# Compiler le code CUDA
mod = SourceModule(open("kernel.cu").read(), options=['-std=c++11'])

# Obtenir la fonction CUDA
dot = mod.get_function("dot_product")
reduce = mod.get_function("reduce")

# Définir les tailles des blocs et des grilles
block_size = 32
n = pow(2,24)  # Taille du tableau
grid_size = (n + block_size - 1) // block_size
print(grid_size)


a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
o = np.zeros(grid_size, dtype=np.float32)


a_gpu = drv.mem_alloc(a.nbytes)
b_gpu = drv.mem_alloc(b.nbytes)
o_gpu = drv.mem_alloc(o.nbytes)


drv.memcpy_htod(a_gpu, a)
drv.memcpy_htod(b_gpu, b)


dot(a_gpu, b_gpu, o_gpu, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1), shared=block_size * a.dtype.itemsize)


drv.memcpy_dtoh(o, o_gpu)

while grid_size > 1:
    n = grid_size
    grid_size = (n + block_size - 1) // block_size
    temp_o = np.zeros(grid_size, dtype=np.float32)
    temp_o_gpu = drv.mem_alloc(temp_o.nbytes)

    reduce(o_gpu, temp_o_gpu, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1), shared=block_size * o.dtype.itemsize)

    drv.memcpy_dtoh(temp_o, temp_o_gpu)
    o = temp_o
    o_gpu = temp_o_gpu

drv.memcpy_dtoh(o, o_gpu)

print("Result :", o[0])
print("Expected :", dot_product(a, b))
