
import numpy as np
import sklearn 
import sklearn.datasets
import sklearn.linear_model
from math import exp,log
import random

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
    W1 =  [[ random.random()-0.5 for _ in range(d_hidden)] for _ in range(d_input)] #Init of the weights between the input layer and the hidden layer (d_input x d_hidden)
    # Bias of the first layer vector of size d_hidden
    b1 = [ random.random()-0.5 for _ in range(d_hidden)] #Init of the biais vector for the hidden layers (1 x d_hidden)
    # Second layer of size d_hidden x d_output
    W2 = [[ random.random()-0.5 for _ in range(d_output)] for _ in range(d_hidden)] #Init of the weights between the hidden layer and the output layer (d_hidden x d_output)
    # The bias of the second layer
    b2 =[ random.random()-0.5 for _ in range(d_output)] #Init of the biais vector of the output layer (1 x d_output)
    # The model returned at the end is a dictionary of weights and biases
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model

######################### FIRST PARALIZATION STEP #########################""
# dot product between two vectors
def dot_product(v1, v2):
    acc=0
    for v_1,v_2 in zip(v1,v2):
        acc+=v_1*v_2
    return acc

# Add two vectors
def add_bias(v1, v2):
    res=[]
    for v_1, v_2 in zip(v1,v2):
        res.append(v_1+v_2) 
    return res
# Get the columns number "index" of W
def get_columns(W, index):
    res=[]
    for row in W:
        res.append(row[index])
    return res

# Transpose a matrix
def transpose(W):
    res=[]
    for i in range(len(W[0])):
        res.append(get_columns(W,i))
    return res

# Multiplication between two matrices()
def matrix_multiplication(X, W):
    res=[]
    for row_x in X:
        tmp=[]
        for i in range(len(W[0])):
            tmp.append(dot_product(row_x,get_columns(W,i))) #Dot product between each row of the matrix X and each line of the matrix W
        res.append(tmp)

    return res


def forward_layer(X, W, b):
    v= matrix_multiplication(X,W)
    for i in range(len(v)):
        v[i]=add_bias(v[i],b) #When X define a set of data points we must add the biais for each data points
    return v
def sigmoid(x):

    return 1 / (1 + np.exp(-1*x)) #Formula of the sigmoid to use as activation function for the hidden layer
def sigmoid_derivative(z):

    return sigmoid(z) * (1 - sigmoid(z)) #Formula of the derivative of the sigmoid to use for the backpropagation
def forward_function(X,W1,b1,W2,b2):     
    #TODO
    z1 = forward_layer(X,W1,b1)  # Output of the first layer 
    a1 = [[sigmoid(z) for z in z_row] for z_row in z1] # Sigmoid activation of the first layer
    z2 = forward_layer(a1,W2,b2)  # Output of the second layer
    exp_scores =[[np.exp(z) for z in z_row] for z_row in z2]# Compute exp(z2)
    probs = [[exp/sum(exp_row) for exp in exp_row] for exp_row in exp_scores] #Apply softmax activation function on z2
    return probs



################# TRAINING FUNCTION ###################

def train_model(model, nn_hdim, num_epochs=1, print_loss=False):

    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']


    # Gradient descent. For each batch...
    for i in range(0, num_epochs):
        
        # Forward propagation (copy/paste inside forward_function previously defined)
        z1 = forward_layer(X, W1, b1)  # Output of the first layer
        a1 = [[sigmoid(z) for z in z_row] for z_row in z1]  # Sigmoid activation of the first layer
        z2 =  forward_layer(a1, W2, b2)  # Output of the second layer
        exp_scores = [[exp(z) for z in z_row] for z_row in z2]# Compute exp(z2)
        probs = [[exp/sum(exp_row) for exp in exp_row] for exp_row in exp_scores] #Compute the softmax function for the output layer
        
        
        correct_logprobs = [log(x_indiv[y_indiv]) for x_indiv, y_indiv in zip(probs, y)]# Calculation of cross entropy for each example
        
        data_loss = -1./N * sum(correct_logprobs) # Loss totale
        
        
        
        # Backpropagation
        #TODO
        #Computing delta2 as the difference between the output of the model and the ground truth
        delta2 = probs
        for output_pred, y_true in zip(delta2, y):
            output_pred[y_true] -= 1
        #Application of the chainning rule 
        dW2 = matrix_multiplication(transpose(a1), delta2)  # Gradient of the weights between the hidden layer and the output layer
        db2 = [sum(d2_row) for d2_row in transpose(delta2)]  # Gradient of the biais of the output layer

        #Computing delta1 from delta2, W2 and z1
        delta1 =matrix_multiplication(delta2, transpose(W2))  
        delta1 = [[sigmoid_derivative(z)*d1 for z,d1 in zip(z1_row,d1_row)] for z1_row, d1_row in zip(z1,delta1)]
        dW1 = matrix_multiplication(transpose(X), delta1)  # Gradient of the weights between the input layer and the hidden layer
        db1 = [sum(d1_row) for d1_row in transpose(delta1)]  # Gradient of the biais of the hidden layer
        
        
        # Gradient descente
        W1 =[[w - epsilon * d for d, w in zip(dW1_row, W1_row)]for dW1_row, W1_row in zip(dW1, W1)]
        b1 = [b - epsilon * d for b, d in zip(b1, db1)]
        W2 = [[w - epsilon * d for d, w in zip(dW1_row, W1_row)]for dW1_row, W1_row in zip(dW2, W2)]
        b2 = [b - epsilon * d for b, d in zip(b2, db2)]
        # Updating weights and biases
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        # Loss display
        if print_loss and i % 50 == 0:
          print("Loss at epoch %i: %f" %(i, data_loss))
      
    return model
######################## PREDICTION FUNCTION ###############
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation, like before
    z1 = forward_layer(x,W1,b1) 
    a1 = [[sigmoid(z) for z in z_row] for z_row in z1]
    z2 = forward_layer(a1,W2,b2) 
    exp_scores =[[np.exp(z) for z in z_row] for z_row in z2]
    probs = [[exp/sum(exp_row) for exp in exp_row] for exp_row in exp_scores]
    return np.argmax(probs, axis=1)

########################## TEST ########################## number of examples in the training set

np.random.seed(1)
X, y = sklearn.datasets.make_moons(300, noise=0.20)

N =  len(X) #TODO size of the dataset

# dimension of the input
d_input = 2 #TODO 2 input features (x,y for each datpoints)

# dimension of the output
d_output = 2 #TODO to final classes for our classification problem

# dimension of the hidden layer i.e. number of neurons in the hidden layer
d_hidden = 32 #TODO 



# learning rate for the gradient descente algorithm
epsilon = 0.01 #TODO

model = init_model(d_input,d_hidden,d_output)
model = train_model(model,d_hidden, num_epochs=500, print_loss=True)
print("The final accuracy obtained is :", accuracy(y, predict(model, X)))