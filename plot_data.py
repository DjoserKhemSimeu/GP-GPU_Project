import matplotlib.pyplot as plt 

def plot_perf():
    # Lire les données des fichiers texte
    with open('data/loop_timeslvl0.txt', 'r') as file:
        loop_lvl0 = file.readlines()
    with open('data/loop_timeslvl2.txt', 'r') as file:
        loop_lvl2 = file.readlines()
    with open('data/epoch_timeslvl0.txt', 'r') as file:
        epoch_lvl0 = file.readlines()
    with open('data/epoch_timeslvl2.txt', 'r') as file:
        epoch_lvl2 = file.readlines()

    # Initialiser les listes pour stocker les valeurs
    loop_lvl0_x = []
    loop_lvl0_y = []
    loop_lvl2_x = []
    loop_lvl2_y = []
    epoch_lvl0_x = []
    epoch_lvl0_y = []
    epoch_lvl2_x = []
    epoch_lvl2_y = []

    # Extraire les valeurs pour les boucles niveau 0
    for line in loop_lvl0:
        parts = line.split()
        if len(parts) == 2:
            loop_lvl0_x.append(float(parts[0]))
            loop_lvl0_y.append(float(parts[1]))

    # Extraire les valeurs pour les boucles niveau 2
    for line in loop_lvl2:
        parts = line.split()
        if len(parts) == 2:
            loop_lvl2_x.append(float(parts[0]))
            loop_lvl2_y.append(float(parts[1]))

    # Extraire les valeurs pour les époques niveau 0
    for line in epoch_lvl0:
        parts = line.split()
        if len(parts) == 2:
            epoch_lvl0_x.append(float(parts[0]))
            epoch_lvl0_y.append(float(parts[1]))

    # Extraire les valeurs pour les époques niveau 2
    for line in epoch_lvl2:
        parts = line.split()
        if len(parts) == 2:
            epoch_lvl2_x.append(float(parts[0]))
            epoch_lvl2_y.append(float(parts[1]))

    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    print(loop_lvl0)
    # Tracer les données des boucles
    ax1.plot(loop_lvl0_x, loop_lvl0_y, marker='o', linestyle='-', color='b', label='Numpy version (CPU)')
    ax1.plot(loop_lvl2_x, loop_lvl2_y, marker='s', linestyle='-', color='r', label='Cuda version (GPU)')
    ax1.set_xlabel('Number of hidden neurons')
    ax1.set_ylabel('Times (s)')
    ax1.set_title('Execution times of the training process')
    ax1.legend()
    ax1.grid(True)

    # Tracer les données des époques
    ax2.plot(epoch_lvl0_x, epoch_lvl0_y, marker='o', linestyle='-', color='b', label='Numpy version (CPU)')
    ax2.plot(epoch_lvl2_x, epoch_lvl2_y, marker='s', linestyle='-', color='r', label='Cuda version (GPU)')
    ax2.set_xlabel('Number of hidden neurons')
    ax2.set_ylabel('Times (s)')
    ax2.set_title('Execution times of a single epoch')
    ax2.legend()
    ax2.grid(True)

    # Ajuster les espaces entre les sous-graphiques
    plt.tight_layout()

    # Afficher les graphiques
    plt.show()
plot_perf()