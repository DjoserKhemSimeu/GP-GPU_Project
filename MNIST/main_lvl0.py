
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
        # start_fw=time.time()
        z1 = forward_layer(X, W1, b1)  # Output of the first layer
        a1 = sigmoid(z1) # Sigmoid activation of the first layer
        z2 =  forward_layer(a1, W2, b2)  # Output of the second layer
        exp_scores = np.exp(z2)# Compute exp(z2)
        probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True) #Compute the softmax function for the output layer
        # end_fw=time.time()
        # if i==0:
        #     print("Time to compute the forward pass =", end_fw-start_fw)
        
        correct_logprobs = np.log(probs[np.arange(probs.shape[0]), y])# Calculation of cross entropy for each example
        
        data_loss = -1./N * np.sum(correct_logprobs,axis=0,keepdims=True) # Loss totale
        
        

        
        
        # Backpropagation
        # start_grad=time.time()
        delta2 = probs.copy()
        delta2[np.arange(probs.shape[0]), y] -= 1
        dW2 = matrix_multiplication(transpose(a1), delta2) 
        db2 = np.sum(delta2, axis=0, keepdims=True) 

        delta1 =sigmoid_derivative(z1) * matrix_multiplication(delta2, transpose(W2))  
        dW1 = matrix_multiplication(transpose(X), delta1) 
        db1 = np.sum(delta1, axis=0, keepdims=True)
        # end_grad=time.time()
        # if i==0:
        #     print("Time to compute the gradients =", end_grad-start_grad)
        
        
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
            with open('../data/epoch_timeslvl0.txt', 'a') as file:
                file.write(f"{K1} {elapsed_time}\n")
        # Loss display
        if print_loss and i % 50 == 0:

          print("Loss at epoch %i: %f" %(i, data_loss))
    end_loop=time.time()
    print("Looping time : ", end_loop-start_loop)
    return model#,history_train,history_val


########################## TEST ########################## number of examples in the training set

digits = sklearn.datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))



X =  digits.images.reshape((n_samples, -1)) # We reshape the images into vector

y = digits.target

N = len(X) 
d_input = 8*8 #TODO The image shape is 8x8 pixels
d_output = 10 #TODO We the classes (numbers from 0 to 9)
#d_hidden = 20 

# Gradient descent parameter
epsilon = 0.001 

for d_hidden in [2,4,8,16,32,64]:
    model = init_model(d_input,d_hidden,d_output)
    start= time.time()
    model = train_model(model,d_hidden, num_epochs=3000, print_loss=False)
    end=time.time()
    print(f"N_Hidden : {d_hidden}   Training time : {end-start}")

print("The final accuracy obtained is :", accuracy(y, predict(model, X)))
