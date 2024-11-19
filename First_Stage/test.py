import numpy as np
from numba import cuda, int32

@cuda.jit()
def add(output):
    sharedMemT = cuda.shared.array(shape=(1), dtype=int32)
    pos = cuda.grid(1)
    if pos == 0:
        sharedMemT[0] = 0

    cuda.syncthreads()

    cuda.atomic.add(sharedMemT, 0, 1)
    cuda.syncthreads()

    if pos == 0:
        output[0] = sharedMemT[0]

out = np.array([0])
add[1, 2](out)
print(out)
