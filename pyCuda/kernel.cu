#define TILE_DIM 16
extern "C" __global__ void dot_product(float *v1, float *v2, float *o, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid]=0;
    __syncthreads();    
    if (i < n) {
        sdata[tid] = v1[i] * v2[i];
    }
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        o[blockIdx.x] = sdata[0];
    }
}


extern "C" __global__ void reduce(float *v1, float *o, int n) {
    extern __shared__ float sdata2[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata2[tid]=0;
    __syncthreads();    
   
    if (i < n) {
        sdata2[tid] = v1[i];
    }
    __syncthreads();

    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata2[tid] += sdata2[tid + s];
        }
        __syncthreads();
    }

    
    if (tid == 0) {
        o[blockIdx.x] = sdata2[0];
    }
}
extern "C" __global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, int BRows,
    int BCols, int CRows, int CCols)
{
    float CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

         if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
             As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)
             Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n)
             CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
}
extern "C" __global__ void transpose(float *in, float *out, unsigned int nx, unsigned int ny){
	unsigned int ix=blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int iy=blockDim.y * blockIdx.y + threadIdx.y;
	if (ix>=nx || iy>=ny) return;
	out[iy*nx + ix]=in[ix*ny + iy];
}