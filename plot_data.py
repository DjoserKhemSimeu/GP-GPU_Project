import matplotlib.pyplot as plt

def plot_perf():
    # Liste des valeurs possibles pour TILE_DIM
    tile_dims = [4, 8, 16, 32]

    # Initialiser les listes pour stocker les valeurs
    loop_lvl0_x = []
    loop_lvl0_y = []
    epoch_lvl0_x = []
    epoch_lvl0_y = []

    # Lire les données des fichiers texte pour le niveau 0
    with open('data/loop_timeslvl00.txt', 'r') as file:
        loop_lvl0 = file.readlines()
    with open('data/epoch_timeslvl00.txt', 'r') as file:
        epoch_lvl0 = file.readlines()

    # Extraire les valeurs pour les boucles niveau 0
    for line in loop_lvl0:
        parts = line.split()
        if len(parts) == 2:
            loop_lvl0_x.append(float(parts[0]))
            loop_lvl0_y.append(float(parts[1]))

    # Extraire les valeurs pour les époques niveau 0
    for line in epoch_lvl0:
        parts = line.split()
        if len(parts) == 2:
            epoch_lvl0_x.append(float(parts[0]))
            epoch_lvl0_y.append(float(parts[1]))

    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Tracer les données des boucles niveau 0
    ax1.plot(loop_lvl0_x, loop_lvl0_y, marker='o', linestyle='-', color='b', label='Numpy version (CPU)')
    ax1.set_xlabel('Number of hidden neurons by hidden layer')
    ax1.set_ylabel('Times (s)')
    ax1.set_title('Execution times of the training process')
    
    ax1.grid(True)

    # Tracer les données des époques niveau 0
    ax2.plot(epoch_lvl0_x, epoch_lvl0_y, marker='o', linestyle='-', color='b', label='Numpy version (CPU)')
    ax2.set_xlabel('Number of hidden neurons by hidden layer')
    ax2.set_ylabel('Times (s)')
    ax2.set_title('Execution times of a single epoch')
    ax2.legend()
    ax2.grid(True)

    # Lire et tracer les données pour chaque valeur de TILE_DIM
    for tile_dim in tile_dims:
        loop_lvl3_x = []
        loop_lvl3_y = []
        epoch_lvl3_x = []
        epoch_lvl3_y = []

        # Lire les données des fichiers texte pour le niveau 3
        with open(f'data/loop_timeslvl3_{tile_dim}.txt', 'r') as file:
            loop_lvl3 = file.readlines()
        with open(f'data/epoch_timeslvl3_{tile_dim}.txt', 'r') as file:
            epoch_lvl3 = file.readlines()

        # Extraire les valeurs pour les boucles niveau 3
        for line in loop_lvl3:
            parts = line.split()
            if len(parts) == 2:
                loop_lvl3_x.append(float(parts[0]))
                loop_lvl3_y.append(float(parts[1]))

        # Extraire les valeurs pour les époques niveau 3
        for line in epoch_lvl3:
            parts = line.split()
            if len(parts) == 2:
                epoch_lvl3_x.append(float(parts[0]))
                epoch_lvl3_y.append(float(parts[1]))

        # Tracer les données des boucles niveau 3
        ax1.plot(loop_lvl3_x, loop_lvl3_y, marker='s', linestyle='-', label=f'Cuda version (GPU) Tile Dim: {tile_dim}')
        ax1.legend()
        # Tracer les données des époques niveau 3
        ax2.plot(epoch_lvl3_x, epoch_lvl3_y, marker='s', linestyle='-', label=f'Cuda version (GPU) Tile Dim: {tile_dim}')
        ax2.legend()
    # Ajuster les espaces entre les sous-graphiques
    plt.tight_layout()

    # Afficher les graphiques
    plt.show()

plot_perf()
