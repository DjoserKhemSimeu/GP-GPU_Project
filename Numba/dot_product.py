from numba import cuda
@cuda.jit
def dot_product(v1, v2,out):
    sdata= cuda.shared.array(shape=(cuda.blockDim.x))
    tid=cuda.threadIdx.x
    i = cuda.blockIdx.x*cuda.blockDim.x+ cuda.threadIdx.x
    sdata[tid]=0
    cuda.synchthreads()
    if(i<n):
        sdata[tid]=v1[i]*v2[i]
    cuda.synchthreads()
    s= cuda.blockDim.x/2
    while s > 0 :
        if tid<s:
            sdata[tid] += sdata[tid+s]
        s/=2
    cuda.syncthreads()
    if tid==0:
        out[cuda.blockIdx.x]=sdata[0]

@cuda.jit
def reduce(v1, out):
    sdata= cuda.shared.array(shape=(cuda.blockDim.x))
    tid=cuda.threadIdx.x
    i = cuda.blockIdx.x*cuda.blockDim.x+ cuda.threadIdx.x
    sdata[tid]=0
    cuda.synchthreads()
    if(i<n):
        sdata[tid]=v1[i]
    cuda.synchthreads()
    s= cuda.blockDim.x/2
    while s > 0 :
        if tid<s:
            sdata[tid] += sdata[tid+s]
        s/=2
    cuda.syncthreads()
    if tid==0:
        out[cuda.blockIdx.x]=sdata[0]

x=[1,2,3,4,5,6,7,8]
y=[1,2,3,4,5,6,7,8]
out=0
dot_product[1,8](x,y,out)
print(out)