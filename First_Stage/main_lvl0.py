
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

def init_model(d_input: int, d_hidden: int, d_output: int):
    """
    Args:
        d_input (int): dimension of the input
        d_hidden (int): dimension of the hidden layer
        d_output (int): dimension of the output

    Returns:
        dict: Dictionary containing 4 keys, the weights/biases (W1,b1) and (W2,b2) of the neural network.
        Each of these weights and biases are lists or list of lists of float.
    """
    # Initialization of random parameters
    random.seed(0)
    # First layer of size d_input x d_hidden
    W1 = np.random.rand(d_input,d_hidden)-0.5 #TODO
    # Bias of the first layer vector of size d_hidden
    b1 = np.random.rand(1,d_hidden)-0.5 #TODO
    # Second layer of size d_hidden x d_output
    W2 = np.random.rand(d_hidden,d_output)-0.5
    # The bias of the second layer
    b2 = np.random.rand(1,d_output)-0.5 #TODO
    # The model returned at the end is a dictionary of weights and biases
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
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
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation, like before
    z1 = forward_layer(x,W1,b1) 
    a1 = sigmoid(z1)
    z2 = forward_layer(a1,W2,b2) 
    exp_scores =np.exp(z2)
    probs = exp_scores/np.sum(exp_scores)
    return np.argmax(probs, axis=1)
def sigmoid(x):

    return 1 / (1 + np.exp(-1*x)) #Formula of the sigmoid to use as activation function for the hidden layer
def sigmoid_derivative(z):

    return sigmoid(z) * (1 - sigmoid(z)) #Formula of the derivative of the sigmoid to use for the backpropagation
def forward_function(x,W1,b1,W2,b2):     
    z1 = forward_layer(x,W1,b1) 
    a1 = sigmoid(z1)
    z2 = forward_layer(a1,W2,b2) 
    exp_scores =np.exp(z2)
    probs = exp_scores/np.sum(exp_scores)
    return probs



################# TRAINING FUNCTION ###################

def train_model(model, nn_hdim, num_epochs=1, print_loss=False):

    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    history_val=[]
    history_train=[]
    
    # Gradient descent. For each batch...
    start_loop=time.time()
    for i in range(0, num_epochs):
        start_e=time.time()
        
        # history_val.append(accuracy(y_val,predict(model,X_val)))
        # history_train.append(accuracy(y,predict(model,X)))

        ##Training
        # Forward propagation (copy/paste inside forward_function previously defined)
        start_fw=time.time()
        z1 = forward_layer(X, W1, b1)  # Output of the first layer
        a1 = sigmoid(z1) # Sigmoid activation of the first layer
        z2 =  forward_layer(a1, W2, b2)  # Output of the second layer
        exp_scores = np.exp(z2)# Compute exp(z2)
        probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True) #Compute the softmax function for the output layer
        end_fw=time.time()
        if i==0:
            print("Time to compute the forward pass =", end_fw-start_fw)
        
        correct_logprobs = np.log(probs[np.arange(probs.shape[0]), y])# Calculation of cross entropy for each example
        
        data_loss = -1./N * np.sum(correct_logprobs,axis=0,keepdims=True) # Loss totale
        
        

        
        
        # Backpropagation
        start_grad=time.time()
        delta2 = probs.copy()
        delta2[np.arange(probs.shape[0]), y] -= 1
        dW2 = matrix_multiplication(transpose(a1), delta2) 
        db2 = np.sum(delta2, axis=0, keepdims=True) 

        delta1 =sigmoid_derivative(z1) * matrix_multiplication(delta2, transpose(W2))  
        dW1 = matrix_multiplication(transpose(X), delta1) 
        db1 = np.sum(delta1, axis=0, keepdims=True)
        end_grad=time.time()
        if i==0:
           print("Time to compute the gradients =", end_grad-start_grad)
        
        
        # Gradient descente
        # start=time.time()
        W1 -= epsilon * dW1
        b1 -= epsilon * db1
        W2 -= epsilon * dW2
        b2 -= epsilon * db2
        # end=time.time()
        # if i==0:
        #     print("Time to update the parameters =", end-start)
        # Updating weights and biases
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        end_e=time.time()
        elapsed_time=end_e-start_e
        
        if i==0:
            print("elapsed time for 1 epoch:", elapsed_time)
            with open('../data/epoch_timeslvl0.txt', 'a') as file:
                file.write(f"{nn_hdim} {elapsed_time}\n")
        # Loss display
        if print_loss and i % 50 == 0:

          print("Loss at epoch %i: %f" %(i, data_loss))
    end_loop=time.time()
    loop_time=end_loop-start_loop
    with open('../data/loop_timeslvl0.txt', 'a') as file:
        file.write(f"{nn_hdim} {loop_time}\n")
    return model#,history_train,history_val


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
X, y = sklearn.datasets.make_moons(600, noise=0.20)
X_save = X
y_save = y

N=len(X)


# dimension of the input
d_input = 2 

# dimension of the output
d_output = 2 

# dimension of the hidden layer i.e. number of neurons in the hidden layer
#d_hidden = 32



# learning rate for the gradient descente algorithm
epsilon = 0.01 
with open('../data/epoch_timeslvl0.txt', 'w') as file:
        pass
with open('../data/loop_timeslvl0.txt', 'w') as file:
    pass
for d_hidden in [2,4,8,16,32,64,128,256,512,1024]:
    model = init_model(d_input,d_hidden,d_output)
    start= time.time()
    model = train_model(model,d_hidden, num_epochs=3000, print_loss=False)
    end=time.time()
    print(f"N_Hidden : {d_hidden}   Training time : {end-start}")

print("The final accuracy obtained is :", accuracy(y, predict(model, X)))
# plot_decision_boundary(lambda x: predict(model, x))
# plt.title("Results by with numpy")
# plt.show()
plot_perf()