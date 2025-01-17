

#include <float.h>

#define TILE_DIM 32
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
extern "C" __global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, int BCols) {
    // Calcul de l'indice de la ligne et de la colonne
    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    // Déclaration de la variable qui va stocker la valeur du produit scalaire
    float CValue = 0.0;

    // Partage des blocs pour les matrices A et B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Boucle sur les "tiles" (blocs de sous-matrices)
    for (int k = 0; k < (ACols + TILE_DIM - 1) / TILE_DIM; ++k) {
        // Charger les sous-matrices A et B dans les mémoires partagées
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

        // Synchroniser les threads avant de commencer les calculs
        __syncthreads();

        // Calcul du produit scalaire pour cette "tile"
        for (int n = 0; n < TILE_DIM; ++n) {
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
        }

        // Synchroniser les threads après le calcul
        __syncthreads();
    }

    // Écrire la valeur calculée dans la matrice C si elle est dans les limites
    if (Row < ARows && Col < BCols) {
        C[Row * BCols + Col] = CValue;
    }
}
extern "C" __global__ void add_bias(float *A, float *B, float *C, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] + B[col];  // Ajout du biais à chaque ligne
    }
}
extern "C" __global__ void transpose(float *in, float *out, unsigned int nx, unsigned int ny) {
    
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if (ix < nx && iy < ny) {
        out[iy*nx+ix]=in[ix*ny+iy];
    }
}

extern "C" __global__ void cross_entropy(float *probs, int *y, float *out, unsigned int n, unsigned int num_classes) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < n) {
        out[ix] = -log(probs[ix * num_classes + y[ix]]);
    }
}

extern "C" __global__ void add (float * WX, float * b, float * out,unsigned int n){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < n) {
        out[ix] = WX[ix]+b[ix];
    }
}
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

extern "C" __global__ void sigmoid_activation(float *A, float *B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        B[row * N + col] = sigmoid(A[row * N + col]);
    }
}

extern "C" __global__ void exp_scores(float *A, float *B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        // Trouver la valeur maximale pour la ligne entière
        __shared__ float max_val[TILE_DIM];
        if (threadIdx.x == 0) {
            float max_tmp = -FLT_MAX;
            for (int i = 0; i < N; ++i) {
                max_tmp = fmaxf(max_tmp, A[row * N + i]);
            }
            max_val[threadIdx.y] = max_tmp;
        }
        __syncthreads();

        // Soustraire max_val et calculer l'exponentielle
        B[row * N + col] = exp(A[row * N + col] - max_val[threadIdx.y]);
    }
}
extern "C" __global__ void softmax(float *A, float *B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // Calculer la somme des exponentielles pour la ligne
        __shared__ float row_sum[TILE_DIM];
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int i = 0; i < N; ++i) {
                sum += exp(A[row * N + i]);
            }
            row_sum[threadIdx.y] = sum;
        }
        __syncthreads();

        // Calculer le softmax
        B[row * N + col] = exp(A[row * N + col]) / row_sum[threadIdx.y];
    }
}
extern "C" __global__ void compute_delta2(float *probs, int *y_true, float *out, unsigned int num_classes, unsigned int n) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < n) {
        for (unsigned int j = 0; j < num_classes; ++j) {
            out[ix * num_classes + j] = probs[ix * num_classes + j];
        }
        
        out[ix * num_classes + y_true[ix]] -= 1.0f;
    }
}
extern "C" __global__ void compute_db(float *delta, float *out, unsigned int n_col, unsigned int n_row,float * b, float epsilon) {
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < n_col) {
        float acc = 0.0f;
        for (unsigned int i = 0; i < n_row; ++i) {
            acc += delta[i * n_col + col];
        }
        // Stockez la somme calculée dans out
        out[col] =b[col]-epsilon*acc;

    }
}
extern "C" __global__ void compute_dW(float* A, float* B, float* C, int ARows, int ACols, int BCols,float * W, float epsilon) {
    // Calcul de l'indice de la ligne et de la colonne
    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    // Déclaration de la variable qui va stocker la valeur du produit scalaire
    float CValue = 0.0;

    // Partage des blocs pour les matrices A et B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Boucle sur les "tiles" (blocs de sous-matrices)
    for (int k = 0; k < (ACols + TILE_DIM - 1) / TILE_DIM; ++k) {
        // Charger les sous-matrices A et B dans les mémoires partagées
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

        // Synchroniser les threads avant de commencer les calculs
        __syncthreads();

        // Calcul du produit scalaire pour cette "tile"
        for (int n = 0; n < TILE_DIM; ++n) {
            CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];
        }

        // Synchroniser les threads après le calcul
        __syncthreads();
    }

    // Écrire la valeur calculée dans la matrice C si elle est dans les limites
    if (Row < ARows && Col < BCols) {
        C[Row * BCols + Col] = W[Row * BCols + Col]-epsilon*CValue;
    }
}
__device__ float sigmoid_derivative(float x) {
    float sigmoid = 1.0f / (1.0f + exp(-x));
    return sigmoid * (1.0f - sigmoid);
}
extern "C" __global__ void compute_delta1(float *delta2, float *z1, float *out, unsigned int n_col, unsigned int n_row) {
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (iy < n_row && ix < n_col) {
        float z = z1[iy * n_col + ix];
        out[iy * n_col + ix] = sigmoid_derivative(z) * delta2[iy * n_col + ix];
    }
}
extern "C" __global__ void update_weights (float * W, float* dW, float * out,float epsilon ,unsigned int n_col,unsigned int n_row){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < n_row) {
        for (unsigned int j = 0; j < n_col; ++j) {
            out[ix * n_col + j] =W[ix * n_col + j] -epsilon * dW[ix * n_col + j] ;
        }
    }
}
extern "C" __global__ void update_bias (float * b, float* db, float * out,float epsilon ,unsigned int n){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix < n) {
        out[ix] = b [ix] - epsilon*db[ix];
    }
}