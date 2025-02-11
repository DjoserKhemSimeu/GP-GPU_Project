import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import time
import os
import sys

# Initialisation de PyCUDA
drv.init()
context = drv.Device(0).make_context()
# Charger le fichier kernel.cu
file_path = os.path.join(os.path.dirname(__file__), 'kernel.cu')
mod = SourceModule(open(file_path).read(), options=['-std=c++11'])
# Obtenir les fonctions du kernel
dot = mod.get_function("dot_product")
reduce = mod.get_function("reduce")
matrix_multiplication = mod.get_function("MatMul")
transpose = mod.get_function("transpose")
cross_entropy = mod.get_function("cross_entropy")
add_bias = mod.get_function("add_bias")
sigmoid_activation = mod.get_function("sigmoid_activation")
exp_scores = mod.get_function("exp_scores")
softmax = mod.get_function("softmax")
compute_delta2 = mod.get_function("compute_delta2")
compute_db = mod.get_function("compute_db")
compute_dW = mod.get_function("compute_dW")
compute_delta1 = mod.get_function("compute_delta1")
update_weights = mod.get_function("update_weights")
update_bias = mod.get_function("update_bias")

# Taille des tuiles pour les opérations matricielles
TILE_DIM = 32

# Fonction pour mesurer le temps d'exécution d'un kernel
def measure_kernel_time(kernel_function, *args, **kwargs):
    start_event = drv.Event()
    end_event = drv.Event()

    start_event.record()
    kernel_function(*args, **kwargs)
    end_event.record()
    end_event.synchronize()

    return start_event.time_till(end_event) * 1e-3  # Convertir en millisecondes

def main():
    float_size = sys.getsizeof(float)
    kernel_times = {}

    # Exemple de données pour les tests
    np.random.seed(0)
    M, N = 256, 256  # Taille des matrices pour les tests
    A = np.random.randn(M, N).astype(np.float32)
    B = np.random.randn(N, M).astype(np.float32)
    C = np.zeros((M, M), dtype=np.float32)
    y = np.random.randint(0, 10, M).astype(np.int32)
    z = np.random.randn(M, N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    delta = np.random.randn(M, N).astype(np.float32)
    epsilon = 0.001

    # Allocation de mémoire GPU
    A_gpu = drv.mem_alloc(A.nbytes)
    B_gpu = drv.mem_alloc(B.nbytes)
    C_gpu = drv.mem_alloc(C.nbytes)
    y_gpu = drv.mem_alloc(y.nbytes)
    z_gpu = drv.mem_alloc(z.nbytes)
    b_gpu = drv.mem_alloc(b.nbytes)
    delta_gpu = drv.mem_alloc(delta.nbytes)

    # Copie des données vers le GPU
    drv.memcpy_htod(A_gpu, A)
    drv.memcpy_htod(B_gpu, B)
    drv.memcpy_htod(y_gpu, y)
    drv.memcpy_htod(z_gpu, z)
    drv.memcpy_htod(b_gpu, b)
    drv.memcpy_htod(delta_gpu, delta)

    # Mesurer le temps d'exécution de chaque kernel
    kernel_times['matrix_multiplication'] = measure_kernel_time(
        matrix_multiplication, A_gpu, B_gpu, C_gpu, np.int32(M), np.int32(N), np.int32(M),
        block=(TILE_DIM, TILE_DIM, 1),
        grid=((M + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM)
    )

    kernel_times['transpose'] = measure_kernel_time(
        transpose, A_gpu, C_gpu, np.int32(M), np.int32(N),
        block=(TILE_DIM, TILE_DIM, 1),
        grid=((N + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM)
    )

    kernel_times['sigmoid_activation'] = measure_kernel_time(
        sigmoid_activation, z_gpu, C_gpu, np.int32(M), np.int32(N),
        block=(TILE_DIM, TILE_DIM, 1),
        grid=((N + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM)
    )

    kernel_times['softmax'] = measure_kernel_time(
        softmax, z_gpu, C_gpu, np.int32(M), np.int32(N),
        block=(TILE_DIM, TILE_DIM, 1),
        grid=((N + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM)
    )

    kernel_times['compute_delta2'] = measure_kernel_time(
        compute_delta2, C_gpu, y_gpu, delta_gpu, np.int32(N), np.int32(M),
        block=(TILE_DIM, 1, 1),
        grid=((M + TILE_DIM - 1) // TILE_DIM, 1)
    )

    kernel_times['compute_db'] = measure_kernel_time(
        compute_db, delta_gpu, b_gpu, np.int32(N), np.int32(M), b_gpu, np.float32(epsilon),
        block=(TILE_DIM, 1, 1),
        grid=((N + TILE_DIM - 1) // TILE_DIM, 1)
    )

    kernel_times['compute_dW'] = measure_kernel_time(
        compute_dW, A_gpu, delta_gpu, C_gpu, np.int32(M), np.int32(N), np.int32(M), A_gpu, np.float32(epsilon),
        block=(TILE_DIM, TILE_DIM, 1),
        grid=((M + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM)
    )

    kernel_times['compute_delta1'] = measure_kernel_time(
        compute_delta1, delta_gpu, z_gpu, C_gpu, np.int32(M), np.int32(N),
        block=(TILE_DIM, TILE_DIM, 1),
        grid=((M + TILE_DIM - 1) // TILE_DIM, (N + TILE_DIM - 1) // TILE_DIM)
    )

    # Écrire les résultats dans un fichier
    with open('kernel_execution_times.txt', 'w') as f:
        for kernel_name, execution_time in kernel_times.items():
            f.write(f"{kernel_name}: {execution_time:.12f} ms\n")

    print("Les temps d'exécution ont été enregistrés dans 'kernel_execution_times.txt'.")

if __name__ == "__main__":
    main()
    context.pop()
