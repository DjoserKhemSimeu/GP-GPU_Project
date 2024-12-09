

extern "C" __global__ void cross_entropy(float **probs, int *y, float *out, unsigned int n) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < n) {
        out[ix] = -log(probs[ix][y[ix]]);
    }
}
