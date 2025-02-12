import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import time
import sys

# Initialisation de PyCUDA
drv.init()

# Créer un contexte GPU
context = drv.Device(0).make_context()

try:
    # Charger le fichier kernel.cu
    file_path = 'kernel.cu'
    mod = SourceModule(open(file_path).read(), options=['-std=c++11'])

    # Obtenir la fonction du kernel add_bias
    add_bias = mod.get_function("add_bias")

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

        # Exemple de données pour les tests
        np.random.seed(0)
        M, N = 256, 256  # Taille des matrices pour les tests
        A = np.random.randn(M, N).astype(np.float32)
        b = np.random.randn(N).astype(np.float32)

        # Allocation de mémoire GPU
        A_gpu = drv.mem_alloc(A.nbytes)
        b_gpu = drv.mem_alloc(b.nbytes)

        # Copie des données vers le GPU
        drv.memcpy_htod(A_gpu, A)
        drv.memcpy_htod(b_gpu, b)

        # Mesurer le temps d'exécution du kernel add_bias
        execution_time = measure_kernel_time(
            add_bias, A_gpu, b_gpu, np.int32(M), np.int32(N),
            block=(TILE_DIM, TILE_DIM, 1),
            grid=((N + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM)
        )

        # Afficher le temps d'exécution
        print(f"Temps d'exécution de add_bias: {execution_time:.6f} ms")

    if __name__ == "__main__":
        main()
finally:
    # Détruire le contexte GPU
    context.pop()
