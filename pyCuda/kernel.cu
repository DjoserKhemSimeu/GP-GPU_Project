

#include <float.h>
// Kernels definiton file


#define TILE_DIM 32


extern "C" __global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, int BCols) {
    // Calculate the row and column index
    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;

    // Declare the variable that will store the dot product value
    float CValue = 0.0;

    // Shared memory for the tiles of matrices A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Loop over the tiles
    for (int k = 0; k < (ACols + TILE_DIM - 1) / TILE_DIM; ++k) {
        // Load the sub-matrices A and B into shared memory
        if (Row < ARows && (k * TILE_DIM + threadIdx.x) < ACols) {
            As[threadIdx.y][threadIdx.x] = A[Row * ACols + k * TILE_DIM + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (Col < BCols && (k * TILE_DIM + threadIdx.y) < ACols) {
            Bs[threadIdx.y][threadIdx.x] = B[(k * TILE_DIM + threadIdx.y) * BCols + Col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Synchronize threads before starting the computation
        __syncthreads();

        // Compute the dot product for this tile
        for (int n = 0; n < TILE_DIM; ++n) {
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
        }

        // Synchronize threads after the computation
        __syncthreads();
    }

    // Write the computed value to matrix C if within bounds
    if (Row < ARows && Col < BCols) {
        C[Row * BCols + Col] = CValue;
    }
}


extern "C" __global__ void add_bias(float *A, float *B, float *C, int M, int N) {
    // Calculate the row and column index for the thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within the bounds of the matrix
    if (row < M && col < N) {
        // Add the bias from vector B to each element in the corresponding row of matrix A
        // Store the result in matrix C
        C[row * N + col] = A[row * N + col] + B[col];
    }
}


extern "C" __global__ void transpose(float *in, float *out, unsigned int nx, unsigned int ny) {
    // Calculate the x and y indices for the thread
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the bounds of the matrix
    if (ix < nx && iy < ny) {
        // Transpose the matrix by swapping the row and column indices
        // Store the transposed element in the output matrix
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}


// Device function to compute the sigmoid of a value
__device__ float sigmoid(float x) {
    // Sigmoid function: 1 / (1 + e^(-x))
    return 1.0f / (1.0f + exp(-x));
}

extern "C" __global__ void sigmoid_activation(float *A, float *B, int M, int N) {
    // Calculate the row and column index for the thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within the bounds of the matrix
    if (row < M && col < N) {
        // Apply the sigmoid function to each element of matrix A
        // Store the result in matrix B
        B[row * N + col] = sigmoid(A[row * N + col]);
    }
}

extern "C" __global__ void softmax(float *A, float *B, int M, int N) {
    // Calculate the row and column index for the thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // Shared memory to store the sum of exponentials for each row
        __shared__ float row_sum[TILE_DIM];

        // Calculate the sum of exponentials for the current row
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int i = 0; i < N; ++i) {
                sum += exp(A[row * N + i]);
            }
            row_sum[threadIdx.y] = sum;
        }

        // Synchronize threads to ensure the sum is calculated before proceeding
        __syncthreads();

        // Compute the softmax for the current element
        B[row * N + col] = exp(A[row * N + col]) / row_sum[threadIdx.y];
    }
}

extern "C" __global__ void compute_delta2(float *probs, int *y_true, float *out, unsigned int num_classes, unsigned int n) {
    // Calculate the index for the thread
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if the thread is within the bounds of the data
    if (ix < n) {
        // Copy the probabilities to the output array
        for (unsigned int j = 0; j < num_classes; ++j) {
            out[ix * num_classes + j] = probs[ix * num_classes + j];
        }

        // Subtract 1 from the probability of the true class
        out[ix * num_classes + y_true[ix]] -= 1.0f;
    }
}


extern "C" __global__ void compute_db(float *delta, float *out, unsigned int n_col, unsigned int n_row, float *b, float epsilon) {
    // Calculate the column index for the thread
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if the thread is within the bounds of the columns
    if (col < n_col) {
        // Initialize an accumulator to sum the delta values for the current column
        float acc = 0.0f;

        // Sum the delta values for the current column
        for (unsigned int i = 0; i < n_row; ++i) {
            acc += delta[i * n_col + col];
        }

        // Store the computed sum in the output array, adjusted by epsilon and the bias term
        out[col] = b[col] - epsilon * acc;
    }
}


extern "C" __global__ void compute_dW(float* A, float* B, float* C, int ARows, int ACols, int BCols, float *W, float epsilon) {
    // Calculate the row and column index
    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;

    // Declare the variable that will store the dot product value
    float CValue = 0.0;

    // Shared memory for the tiles of matrices A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Loop over the tiles
    for (int k = 0; k < (ACols + TILE_DIM - 1) / TILE_DIM; ++k) {
        // Load the sub-matrices A and B into shared memory
        if (Row < ARows && (k * TILE_DIM + threadIdx.x) < ACols) {
            As[threadIdx.y][threadIdx.x] = A[Row * ACols + k * TILE_DIM + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (Col < BCols && (k * TILE_DIM + threadIdx.y) < ACols) {
            Bs[threadIdx.y][threadIdx.x] = B[(k * TILE_DIM + threadIdx.y) * BCols + Col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Synchronize threads before starting the computation
        __syncthreads();

        // Compute the dot product for this tile
        for (int n = 0; n < TILE_DIM; ++n) {
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
        }

        // Synchronize threads after the computation
        __syncthreads();
    }

    // Write the computed value to matrix C if within bounds
    if (Row < ARows && Col < BCols) {
        // Update the weight matrix W with the gradient adjustment
        C[Row * BCols + Col] = W[Row * BCols + Col] - epsilon * CValue;
    }
}


// Device function to compute the derivative of the sigmoid function
__device__ float sigmoid_derivative(float x) {
    // Compute the sigmoid of x
    float sigmoid = 1.0f / (1.0f + exp(-x));
    // Return the derivative of the sigmoid function
    return sigmoid * (1.0f - sigmoid);
}

extern "C" __global__ void compute_delta1(float *delta2, float *z1, float *out, unsigned int n_col, unsigned int n_row) {
    // Calculate the row and column index for the thread
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    // Check if the thread is within the bounds of the matrix
    if (iy < n_row && ix < n_col) {
        // Retrieve the corresponding element from z1
        float z = z1[iy * n_col + ix];
        // Compute the delta value using the sigmoid derivative and delta2
        out[iy * n_col + ix] = sigmoid_derivative(z) * delta2[iy * n_col + ix];
    }
}
