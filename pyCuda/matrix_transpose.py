import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

TILE_DIM = 4
N = 4
idata = np.random.randn(N, N).astype(np.float32)
odata = np.zeros(N,N)

idata_gpu = drv.mem_alloc(idata.nbytes)
odata_gpu = drv.mem_alloc(odata.nbytes)

drv.memcpy_htod(idata_gpu, idata)

block_dim = (TILE_DIM, TILE_DIM, 1)
grid_dim = (N // TILE_DIM, N // TILE_DIM, 1)

mod = SourceModule(open("kernel.cu").read(), options=['-std=c++11'])

transpose= mod.get_function("transpose")

transpose(odata_gpu, idata_gpu,N,N, block=block_dim, grid=grid_dim)

drv.memcpy_dtoh(odata, odata_gpu)


print("Input matrix:")
print(idata)
print("Transposed matrix:")
print(odata)