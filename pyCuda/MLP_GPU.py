
import numpy as np
import sklearn 
import sklearn.datasets
import sklearn.linear_model
from math import exp,log
import random
import pycuda.autoinit
import time
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt 
import os
import sys
drv.init()
file_path = os.path.join(os.path.dirname(__file__), 'kernel.cu')
mod = SourceModule(open(file_path).read(), options=['-std=c++11'])

matrix_multiplication = mod.get_function("MatMul")
transpose= mod.get_function("transpose")
add_bias = mod.get_function("add_bias")
sigmoid_activation = mod.get_function("sigmoid_activation")
softmax = mod.get_function("softmax")
compute_delta2=mod.get_function("compute_delta2")
compute_db=mod.get_function("compute_db")
compute_dW=mod.get_function("compute_dW")
compute_delta1=mod.get_function("compute_delta1")

TILE_DIM = 32


def accuracy(y_true, y_pred):
    """
    Args:
        y_true (list[int]): A list of integers having values in {0,1} that contain the class labels
        y_pred (list[int]): A list of integers having values in {0,1} that contain the predictions of the model

    Returns:
        float: The Accuracy of the model
    
    Example:
    >>> accuracy([0,0,1], [0,1,1])
    0.666...
    """
    acc=0 #Defining an accumulator
    # print("GT : ",y_true)
    # print("Pred : ",y_pred)
    # print("Diff : ",y_pred-y_true)
    for y_t,y_p in zip(y_true,y_pred):
        if y_t == y_p:
            acc+=1 #Increment it for each good prediction
    return acc/len(y_true) #Return the number of good prediction divided by the total number of predictions


# MODEL CLASS
class Model :
    def __init__(self, nb_hidden, param):
        self.nb_hidden = nb_hidden
        self.param = param



def init_model(d_input: int, d_hidden: list, d_output: int):
    """
    Initialize a neural network model with random weights and biases.

    Args:
        d_input (int): Dimension of the input layer.
        d_hidden (list): List of dimensions for each hidden layer.
        d_output (int): Dimension of the output layer.

    Returns:
        Model: An instance of the Model class containing the initialized weights and biases.
    """
    # Set the random seed for reproducibility
    random.seed(0)

    # Lists to store weights and biases for both CPU (NumPy) and GPU
    weights_np = []
    weights_gpu = []
    biases_np = []
    biases_gpu = []
    W_shapes = []
    b_shapes = []

    # Initialize the first layer weights and biases
    W1 = np.random.rand(d_input, d_hidden[0]) - 0.5
    W1 = np.ascontiguousarray(W1, dtype=np.float32)
    b1 = np.random.rand(1, d_hidden[0]) - 0.5
    b1 = np.ascontiguousarray(b1, dtype=np.float32)

    # Allocate memory on the GPU and copy the weights and biases
    W1_gpu = drv.mem_alloc(W1.nbytes)
    b1_gpu = drv.mem_alloc(b1.nbytes)
    drv.memcpy_htod(W1_gpu, W1)
    drv.memcpy_htod(b1_gpu, b1)

    # Store the weights and biases in lists
    weights_np.append(W1)
    biases_np.append(b1)
    weights_gpu.append(W1_gpu)
    biases_gpu.append(b1_gpu)
    W_shapes.append(W1.shape)
    b_shapes.append(b1.shape)

    # Initialize weights and biases for each subsequent hidden layer
    for i in range(1, len(d_hidden)):
        Wi = np.random.rand(d_hidden[i-1], d_hidden[i]) - 0.5
        Wi = np.ascontiguousarray(Wi, dtype=np.float32)
        bi = np.random.rand(1, d_hidden[i]) - 0.5
        bi = np.ascontiguousarray(bi, dtype=np.float32)

        # Allocate memory on the GPU and copy the weights and biases
        Wi_gpu = drv.mem_alloc(Wi.nbytes)
        bi_gpu = drv.mem_alloc(bi.nbytes)
        drv.memcpy_htod(Wi_gpu, Wi)
        drv.memcpy_htod(bi_gpu, bi)

        # Store the weights and biases in lists
        weights_np.append(Wi)
        biases_np.append(bi)
        weights_gpu.append(Wi_gpu)
        biases_gpu.append(bi_gpu)
        W_shapes.append(Wi.shape)
        b_shapes.append(bi.shape)

    # Initialize the output layer weights and biases
    W_out = np.random.rand(d_hidden[-1], d_output) - 0.5
    W_out = np.ascontiguousarray(W_out, dtype=np.float32)
    b_out = np.random.rand(1, d_output) - 0.5
    b_out = np.ascontiguousarray(b_out, dtype=np.float32)

    # Allocate memory on the GPU and copy the weights and biases
    W_out_gpu = drv.mem_alloc(W_out.nbytes)
    b_out_gpu = drv.mem_alloc(b_out.nbytes)
    drv.memcpy_htod(W_out_gpu, W_out)
    drv.memcpy_htod(b_out_gpu, b_out)

    # Store the weights and biases in lists
    weights_np.append(W_out)
    biases_np.append(b_out)
    weights_gpu.append(W_out_gpu)
    biases_gpu.append(b_out_gpu)
    W_shapes.append(W_out.shape)
    b_shapes.append(b_out.shape)

    # Create a dictionary to store the model parameters
    model_params = {
        'W_np': weights_np,
        'W_gpu': weights_gpu,
        'b_np': biases_np,
        'b_gpu': biases_gpu,
        'W_shapes': W_shapes,
        'b_shapes': b_shapes
    }

    # Initialize and return a Model instance with the parameters
    model = Model(nb_hidden=len(d_hidden), param=model_params)
    return model


def forward_layer(X_gpu, W_gpu, b_gpu, M, N, K):
    """
    Perform a forward pass through a layer of a neural network using GPU.

    Args:
        X_gpu (int): Pointer to the input matrix on the GPU.
        W_gpu (int): Pointer to the weight matrix on the GPU.
        b_gpu (int): Pointer to the bias vector on the GPU.
        M (int): Number of rows in the input matrix.
        N (int): Number of columns in the input matrix (and rows in the weight matrix).
        K (int): Number of columns in the weight matrix (and output matrix).

    Returns:
        int: Pointer to the output matrix on the GPU.
    """
    # Allocate memory on the GPU for the output matrix
    C_gpu = drv.mem_alloc(M * K * np.float32().itemsize)

    # Define the block and grid size for the matrix multiplication kernel
    block_size = (TILE_DIM, TILE_DIM, 1)
    grid_size = ((K + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM)

    # Launch the matrix multiplication kernel
    matrix_multiplication(X_gpu, W_gpu, C_gpu, np.int32(M), np.int32(N), np.int32(K), block=block_size, grid=grid_size)

    # Launch the bias addition kernel
    add_bias(C_gpu, b_gpu, C_gpu, np.int32(M), np.int32(K), block=block_size, grid=grid_size)

    # Return the pointer to the output matrix on the GPU
    return C_gpu

def sigmoid(x):
    """
    Fonction sigmoïde.
    :param x: Entrée (peut être un scalaire, un vecteur ou une matrice)
    :return: La valeur de la sigmoïde appliquée à x
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Fonction dérivée de la sigmoïde.
    :param x: Entrée (peut être un scalaire, un vecteur ou une matrice)
    :return: La dérivée de la sigmoïde appliquée à x
    """
    s = sigmoid(x)
    return s * (1 - s)

def softmax_cpu(x):
    """
    Compute the softmax of a matrix using NumPy on the CPU.

    Args:
        x (numpy.ndarray): Input matrix of shape (n_samples, n_features).

    Returns:
        numpy.ndarray: Output matrix with the softmax applied to each row.
    """
    # Compute the exponential of each element in the input matrix
    exp_scores = np.exp(x)

    # Compute the softmax by dividing each element by the row sum of exponentials
    # The keepdims=True parameter ensures that the result keeps the same dimensions as the input
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


################# TRAINING FUNCTION ###################

def train_model(model, nn_hdim, num_epochs=1, epsilon=0.001, print_loss=False):
    """
    Train a Multi-Layer Perceptron using gradient descent on the GPU.

    Args:
        model (Model): The neural network model to train.
        nn_hdim (list): List of hidden layer dimensions.
        num_epochs (int): Number of epochs to train.
        epsilon (float): Learning rate.
        print_loss (bool): Whether to print the loss during training.

    Returns:
        Model: The trained neural network model.
    """
    params = model.param
    float_size = sys.getsizeof(float)

    start_loop = time.time()

    # Gradient descent loop
    for i in range(num_epochs):
        start = time.time()

        # Allocate memory on the GPU for input and labels
        X_gpu = drv.mem_alloc(X.nbytes)
        drv.memcpy_htod(X_gpu, X)
        y_gpu = drv.mem_alloc(y.nbytes)
        drv.memcpy_htod(y_gpu, y)

        M, N = X.shape
        K_out = params['W_shapes'][-1][1]

        activations = [X_gpu]
        activations_cpu = [X]
        zs = []

        # Forward pass
        for j, (w, b) in enumerate(zip(params["W_gpu"], params["b_gpu"])):
            K0 = params['W_shapes'][j][0]
            K1 = params['W_shapes'][j][1]

            z = forward_layer(activations[-1], w, b, M, K0, K1)
            zs.append(z)

            act_gpu = drv.mem_alloc(M * K1 * float_size)
            if j < len(params["W_gpu"]) - 1:  # Apply sigmoid activation for hidden layers
                sigmoid_activation(z, act_gpu, np.int32(M), np.int32(K1),
                                   block=(TILE_DIM, TILE_DIM, 1),
                                   grid=((K1 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
            else:  # Apply softmax activation for the output layer
                softmax(z, act_gpu, np.int32(M), np.int32(K1),
                        block=(TILE_DIM, TILE_DIM, 1),
                        grid=((K1 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))

            activations.append(act_gpu)

        probs_gpu = activations[-1]
        probs = np.zeros((M, K_out), dtype=np.float32)
        drv.memcpy_dtoh(probs, probs_gpu)

        # Calculate cross-entropy loss
        correct_logprobs = np.log(probs[np.arange(probs.shape[0]), y])
        data_loss = -1. / N * np.sum(correct_logprobs, axis=0, keepdims=True)

        # Backpropagation
        grid_size = (probs.shape[0] + TILE_DIM - 1) // TILE_DIM
        delta2_gpu = drv.mem_alloc(probs.nbytes)
        compute_delta2(probs_gpu, y_gpu, delta2_gpu, np.int32(probs.shape[1]), np.int32(probs.shape[0]),
                       block=(TILE_DIM, 1, 1),
                       grid=(grid_size, 1))
        deltas = [delta2_gpu]

        # Compute deltas for hidden layers
        for j in range(len(params["W_gpu"]) - 2, -1, -1):
            K0 = params['W_shapes'][j][0]
            K1 = params['W_shapes'][j + 1][0]
            K2 = params['W_shapes'][j + 1][1]

            Wi = params["W_gpu"][j]
            Wii = params["W_gpu"][j + 1]

            Wii_T = drv.mem_alloc(params['W_np'][j + 1].nbytes)
            grid_y = (K2 + TILE_DIM - 1) // TILE_DIM
            grid_x = (K1 + TILE_DIM - 1) // TILE_DIM
            transpose(Wii, Wii_T, np.int32(K1), np.int32(K2),
                      block=(TILE_DIM, TILE_DIM, 1),
                      grid=(grid_x, grid_y))

            delta_dot_gpu = drv.mem_alloc(M * K1 * float_size)
            matrix_multiplication(deltas[len(params["W_gpu"]) - 2 - j], Wii_T, delta_dot_gpu, np.int32(M), np.int32(K2), np.int32(K1),
                                  block=(TILE_DIM, TILE_DIM, 1),
                                  grid=((K1 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))

            delta_gpu = drv.mem_alloc(M * K1 * float_size)
            compute_delta1(delta_dot_gpu, zs[j], delta_gpu, np.int32(M), np.int32(K1),
                           block=(TILE_DIM, TILE_DIM, 1),
                           grid=((M + TILE_DIM - 1) // TILE_DIM, (K1 + TILE_DIM - 1) // TILE_DIM))
            deltas.append(delta_gpu)

        deltas.reverse()

        # Update weights and biases
        for j in range(len(params["W_gpu"])):
            K0 = params['W_shapes'][j][0]
            K1 = params['W_shapes'][j][1]

            act_gpu_T = drv.mem_alloc(M * K0 * float_size)
            grid_y = (K0 + TILE_DIM - 1) // TILE_DIM
            grid_x = (M + TILE_DIM - 1) // TILE_DIM
            transpose(activations[j], act_gpu_T, np.int32(M), np.int32(K0),
                       block=(TILE_DIM, TILE_DIM, 1),
                       grid=(grid_x, grid_y))

            dW_gpu = drv.mem_alloc(K0 * K1 * float_size)
            compute_dW(act_gpu_T, deltas[j], dW_gpu, np.int32(K0), np.int32(M), np.int32(K1), params['W_gpu'][j], np.float32(epsilon),
                        block=(TILE_DIM, TILE_DIM, 1),
                        grid=((K1 + TILE_DIM - 1) // TILE_DIM, (K0 + TILE_DIM - 1) // TILE_DIM))
            params["W_gpu"][j] = dW_gpu

            db_gpu = drv.mem_alloc(K1 * float_size)
            compute_db(deltas[j], db_gpu, np.int32(K1), np.int32(M), params["b_gpu"][j], np.float32(epsilon),
                        block=(TILE_DIM, 1, 1),
                        grid=((K1 + TILE_DIM - 1) // TILE_DIM, 1))
            params["b_gpu"][j] = db_gpu

        end = time.time()
        elapsed_time = end - start

        if print_loss and i % 1 == 0:
            print("Loss at epoch %i: %f" % (i, data_loss))

    end_loop = time.time()
    loop_time = end_loop - start_loop

    # Copy the updated weights and biases back to the CPU
    for i in range(len(params['W_gpu'])):
        W_h = np.zeros_like(params['W_np'][i])
        drv.memcpy_dtoh(W_h, params['W_gpu'][i])
        params['W_np'][i] = W_h

        b_h = np.zeros_like(params['b_np'][i])
        drv.memcpy_dtoh(b_h, params['b_gpu'][i])
        params['b_np'][i] = b_h

    return model



########################## TEST #########################


def forward_layer_cpu(X, W, b):
    """
    Perform a forward pass through a layer of a neural network on the CPU.

    Args:
        X (numpy.ndarray): Input matrix of shape (M, N).
        W (numpy.ndarray): Weight matrix of shape (N, K).
        b (numpy.ndarray): Bias vector of shape (1, K).

    Returns:
        numpy.ndarray: Output matrix of shape (M, K) after applying weights and biases.
    """
    # Compute the dot product of the input matrix and the weight matrix
    v = np.dot(X, W)

    # Add the bias vector to the result
    return v + b

def predict_cpu(model, x):
    """
    Make predictions using a neural network model on the CPU.

    Args:
        model (Model): The trained neural network model.
        x (numpy.ndarray): Input data for which predictions are to be made.

    Returns:
        numpy.ndarray: Predicted class labels for the input data.
    """
    params = model.param

    # Initialize the activations list with the input data
    activations = [x]

    # Iterate over each layer in the model
    for j, (w, b) in enumerate(zip(params["W_np"], params["b_np"])):
        K0 = params['W_shapes'][j][0]
        K1 = params['W_shapes'][j][1]

        # Perform the forward pass for the current layer
        z = forward_layer_cpu(activations[-1], w, b)

        # Apply the sigmoid activation function for hidden layers
        if j < len(params["W_np"]) - 1:
            act = sigmoid(z)
        else:
            # Apply the softmax activation function for the output layer
            act = softmax_cpu(z)

        # Append the activated output to the activations list
        activations.append(act)

    # Get the probabilities from the last activation (output layer)
    probs = activations[-1]

    # Return the predicted class labels by taking the argmax along the last axis
    return np.argmax(probs, axis=1)



# Set the random seed for reproducibility
np.random.seed(1)

# Generate a synthetic dataset using make_moons
X, y = sklearn.datasets.make_moons(600, noise=0.1)
X = np.ascontiguousarray(X, dtype=np.float32)  # Ensure the input data is contiguous and of type float32
y = np.ascontiguousarray(y, dtype=np.int32)    # Ensure the labels are contiguous and of type int32

N = len(X)  # Number of samples in the dataset

# Dimension of the input features
d_input = 2  # Each data point has 2 features (x, y coordinates)

# Dimension of the output (number of classes)
d_output = 2  # Binary classification problem with 2 classes


# Iterate over different hidden layer sizes to test model performance
for h in [256]:
    d_hidden = [h] * 4  # Set each hidden layer to have 'h' neurons
    model = init_model(d_input, d_hidden, d_output)  # Initialize the model with the specified architecture

    start = time.time()  # Record the start time for training
    model = train_model(model, d_hidden, num_epochs=1000, print_loss=True)  # Train the model
    end = time.time()  # Record the end time for training

    # Print the training time and hidden layer size
    print(f"N_Hidden : {d_hidden[0]}   Training time : {end - start}")

    # Calculate and print the final accuracy of the model
    print("The final accuracy obtained is :", accuracy(y, predict_cpu(model, X)))

    # Free the GPU memory allocated for the model parameters
    for j in range(len(model.param["W_gpu"])):
        model.param["W_gpu"][j].free()
        model.param["b_gpu"][j].free()

#### display function
def plot_decision_boundary(pred_func):
    """
    Shows the decision boundaries of a binary prediction function.
    """
    # Set grid dimensions and give some margin for display
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    # Generate the grid of points with a distance of h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Drawing the decision boundary
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Show contour and training points
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    
plot_decision_boundary(lambda v: predict_cpu(model,v))
plt.title("Decision Boundary")
plt.show()
