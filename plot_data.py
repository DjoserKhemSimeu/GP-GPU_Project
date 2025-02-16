import matplotlib.pyplot as plt

def plot_perf():
    """
    Plot the performance of training times for different configurations.
    """
    # List of possible values for TILE_DIM
    tile_dims = [4, 8, 16, 32]

    # Initialize lists to store values
    loop_lvl0_x = []
    loop_lvl0_y = []
    epoch_lvl0_x = []
    epoch_lvl0_y = []

    # Read data from text files for level 0
    with open('data/loop_timeslvl00.txt', 'r') as file:
        loop_lvl0 = file.readlines()
    with open('data/epoch_timeslvl00.txt', 'r') as file:
        epoch_lvl0 = file.readlines()

    # Extract values for loop times at level 0
    for line in loop_lvl0:
        parts = line.split()
        if len(parts) == 2:
            loop_lvl0_x.append(float(parts[0]))
            loop_lvl0_y.append(float(parts[1]))

    # Extract values for epoch times at level 0
    for line in epoch_lvl0:
        parts = line.split()
        if len(parts) == 2:
            epoch_lvl0_x.append(float(parts[0]))
            epoch_lvl0_y.append(float(parts[1]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot loop times data for level 0
    ax1.plot(loop_lvl0_x, loop_lvl0_y, marker='o', linestyle='-', color='b', label='Numpy version (CPU)')
    ax1.set_xlabel('Number of hidden neurons by hidden layer')
    ax1.set_ylabel('Times (s)')
    ax1.set_title('Execution times of the training process')
    ax1.grid(True)

    # Plot epoch times data for level 0
    ax2.plot(epoch_lvl0_x, epoch_lvl0_y, marker='o', linestyle='-', color='b', label='Numpy version (CPU)')
    ax2.set_xlabel('Number of hidden neurons by hidden layer')
    ax2.set_ylabel('Times (s)')
    ax2.set_title('Execution times of a single epoch')
    ax2.legend()
    ax2.grid(True)

    # Read and plot data for each TILE_DIM value
    for tile_dim in tile_dims:
        loop_lvl3_x = []
        loop_lvl3_y = []
        epoch_lvl3_x = []
        epoch_lvl3_y = []

        # Read data from text files for level 3
        with open(f'data/loop_timeslvl3_{tile_dim}.txt', 'r') as file:
            loop_lvl3 = file.readlines()
        with open(f'data/epoch_timeslvl3_{tile_dim}.txt', 'r') as file:
            epoch_lvl3 = file.readlines()

        # Extract values for loop times at level 3
        for line in loop_lvl3:
            parts = line.split()
            if len(parts) == 2:
                loop_lvl3_x.append(float(parts[0]))
                loop_lvl3_y.append(float(parts[1]))

        # Extract values for epoch times at level 3
        for line in epoch_lvl3:
            parts = line.split()
            if len(parts) == 2:
                epoch_lvl3_x.append(float(parts[0]))
                epoch_lvl3_y.append(float(parts[1]))

        # Plot loop times data for level 3
        ax1.plot(loop_lvl3_x, loop_lvl3_y, marker='s', linestyle='-', label=f'Cuda version (GPU) Tile Dim: {tile_dim}')
        ax1.legend()

        # Plot epoch times data for level 3
        ax2.plot(epoch_lvl3_x, epoch_lvl3_y, marker='s', linestyle='-', label=f'Cuda version (GPU) Tile Dim: {tile_dim}')
        ax2.legend()


    plt.tight_layout()
    plt.show()

plot_perf()
