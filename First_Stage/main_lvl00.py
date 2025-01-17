
import numpy as np
import sklearn 
import sklearn.datasets
import sklearn.linear_model
from math import exp,log
import random
import time
import matplotlib.pyplot as plt

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
    for y_t,y_p in zip(y_true,y_pred):
        if y_t == y_p:
            acc+=1 #Increment it for each good prediction
    return acc/len(y_true) #Return the number of good prediction divided by the total number of predictions
class Model :
    def __init__(self, nb_hidden, param):
        self.nb_hidden = nb_hidden
        self.param = param

def init_model(d_input: int, d_hidden: list, d_output: int):
    """
    Args:
        d_input (int): dimension of the input
        d_hidden (list): list of dimensions of the hidden layers
        d_output (int): dimension of the output

    Returns:
        Model: An instance of the Model class containing the weights and biases of the MLP.
    """

    random.seed(0)
    
    weights = []
    biases = []
    
    W1 = np.random.rand(d_input, d_hidden[0]) - 0.5
    b1 = np.random.rand(1, d_hidden[0]) - 0.5
    weights.append(W1)
    biases.append(b1)
    
    for i in range(1, len(d_hidden)):
        Wi = np.random.rand(d_hidden[i-1], d_hidden[i]) - 0.5
        bi = np.random.rand(1, d_hidden[i]) - 0.5
        weights.append(Wi)
        biases.append(bi)

    W_out = np.random.rand(d_hidden[-1], d_output) - 0.5
    b_out = np.random.rand(1, d_output) - 0.5
    weights.append(W_out)
    biases.append(b_out)
    
    model_params = {
        'W': weights,
        'b': biases
    }

    model = Model(nb_hidden=len(d_hidden), param=model_params)
    return model
######################### FIRST PARALIZATION STEP #########################""
# dot product between two vectors
def dot_product(v1, v2):
    
    return np.dot(v1,v2)

# Add two vectors
def add_bias(v1, v2):
    return v1+v2
# Get the columns number "index" of W
def get_columns(W, index):
    return W[:,index]

# Transpose a matrix
def transpose(W):
   
    return W.T

# Multiplication between two matrices()
def matrix_multiplication(X, W):
 

    return np.dot(X, W)


def forward_layer(X, W, b):
    v= matrix_multiplication(X,W)
    return add_bias(v,b)
def predict(model, x):
    probs = forward_function(x,model.param)
    return np.argmax(probs, axis=1)
def softmax(x):
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
def sigmoid(x):

    return 1 / (1 + np.exp(-1*x)) #Formula of the sigmoid to use as activation function for the hidden layer
def sigmoid_derivative(z):

    return sigmoid(z) * (1 - sigmoid(z)) #Formula of the derivative of the sigmoid to use for the backpropagation
def forward_function(X,params):     
    for i, (w, b) in enumerate(zip(params["W"], params["b"])): # iterate on all the layers of the model
        X = forward_layer(X, w, b)
        if i < len(params["W"]) - 1:  # Apply the sigmoid for all the hidden layers
            X = sigmoid(X)
      
    exp_scores = np.exp(X)# Compute exp(z2)
    probs = exp_scores/np.sum(exp_scores) #Apply softmax activation function 
    return probs


################# TRAINING FUNCTION ###################

def train_model(model, X, y,h, num_epochs=1, print_loss=False, epsilon=0.01):
    params = model.param
   
    # Gradient descent. For each batch...
    start_loop=time.time()
    for i in range(num_epochs):
        start_e=time.time()
        

        
        # Forward propagation
        activations = [X] # We consider the first activation as the input layer
        zs = []
        for j, (w, b) in enumerate(zip(params["W"], params["b"])):
            z = forward_layer(activations[-1], w, b)
            zs.append(z)
            if j < len(params["W"]) - 1:  # Apply the sigmoid on the hidden layer
                activation = sigmoid(z)
            else: # Apply the softmax function on the output layer
                activation = softmax(z)
            activations.append(activation)

        probs = activations[-1]
        

        # Estimate the loss (c)
        correct_logprobs = np.log(probs[range(len(y)), y])
        data_loss = -np.mean(correct_logprobs)
        # Adding to the loss function the magnitude of the weights of the model (weight decay)
        

        # Backpropagation 
        deltas = [probs] #The delta of the last layer is the difference between the prediction and the target
        deltas[-1][range(len(y)), y] -= 1

        for j in range(len(params["W"])-2, -1, -1):
            
            delta = sigmoid_derivative(zs[j]) * np.dot(deltas[len(params["W"])-2-j],params["W"][j+1].T)

            deltas.append(delta)
        
        deltas.reverse()
        

        # Gradient descent
        for j in range(len(params["W"])):
            
            params["W"][j] -= epsilon * (np.dot(activations[j].T, deltas[j]))
           
            params["b"][j] -= epsilon * (np.sum(deltas[j], axis=0, keepdims=True) )

        # Loss display
        if print_loss and i % 50 == 0:
            print("Loss at epoch %i: %f" % (i, data_loss))
        end_e=time.time()
        elapsed_time=end_e-start_e
        
        if i==0:
            print("elapsed time for 1 epoch:", elapsed_time)
            with open('../data/epoch_timeslvl00.txt', 'a') as file:
                file.write(f"{h} {elapsed_time}\n")
    end_loop=time.time()
    loop_time=end_loop-start_loop
    with open('../data/loop_timeslvl00.txt', 'a') as file:
        file.write(f"{h} {loop_time}\n")
    return model#, history_train, history_val


########################## TEST ########################## number of examples in the training set

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
def plot_perf():
    # Lire les données des fichiers texte
    with open('../data/loop_timeslvl0.txt', 'r') as file:
        loop_lvl0 = file.readlines()
    with open('../data/loop_timeslvl2.txt', 'r') as file:
        loop_lvl2 = file.readlines()
    with open('../data/epoch_timeslvl0.txt', 'r') as file:
        epoch_lvl0 = file.readlines()
    with open('../data/epoch_timeslvl2.txt', 'r') as file:
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
    ax1.plot(loop_lvl0_x, loop_lvl0_y, marker='o', linestyle='-', color='b', label='Level 0')
    ax1.plot(loop_lvl2_x, loop_lvl2_y, marker='s', linestyle='-', color='r', label='Level 2')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.set_title('Comparaison des boucles (Loop)')
    ax1.legend()
    ax1.grid(True)

    # Tracer les données des époques
    ax2.plot(epoch_lvl0_x, epoch_lvl0_y, marker='o', linestyle='-', color='b', label='Level 0')
    ax2.plot(epoch_lvl2_x, epoch_lvl2_y, marker='s', linestyle='-', color='r', label='Level 2')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    ax2.set_title('Comparaison des époques (Epoch)')
    ax2.legend()
    ax2.grid(True)

    # Ajuster les espaces entre les sous-graphiques
    plt.tight_layout()

    # Afficher les graphiques
    plt.show()

np.random.seed(1)
X, y = sklearn.datasets.make_moons(600, noise=0.1)
X_save = X
y_save = y

# Create a permutation of indices
#np.random.permutation(len(X))

# Shuffle X and y using the permutation

# Split the shuffled data into training and validation sets


# dimension of the hidden layer i.e. number of neurons in the hidden layer
#d_hidden = 32



# learning rate for the gradient descente algorithm
epsilon = 0.01 
# with open('../data/epoch_timeslvl0.txt', 'w') as file:
#         pass
# with open('../data/loop_timeslvl0.txt', 'w') as file:
#     pass
d_input = 2

d_output = 2
with open('../data/epoch_timeslvl00.txt', 'w') as file:
        pass
with open('../data/loop_timeslvl00.txt', 'w') as file:
    pass
for h in [2,4,8,16,32,64,128,254,512,1024]:
    d_hidden = [h]*4 

    model = init_model(d_input, d_hidden, d_output)
    start= time.time()
    model = train_model(model,X,y,h, num_epochs=1000,epsilon=0.01, print_loss=False)
    end=time.time()
    print(f"N_Hidden : {h}   Training time : {end-start}")
    print("The final accuracy obtained is :", accuracy(y, predict(model, X)))


plot_decision_boundary(lambda x: predict(model, x))
plt.title("Results by with numpy")
plt.show()
