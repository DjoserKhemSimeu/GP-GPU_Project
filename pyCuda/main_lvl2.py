
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
import sys
drv.init()

mod = SourceModule(open("kernel.cu").read(), options=['-std=c++11'])

dot = mod.get_function("dot_product")
reduce = mod.get_function("reduce")
matrix_multiplication = mod.get_function("MatMul")
transpose= mod.get_function("transpose")
cross_entropy= mod.get_function("cross_entropy")
add_bias = mod.get_function("add_bias")
sigmoid_activation = mod.get_function("sigmoid_activation")
exp_scores=mod.get_function("exp_scores")
softmax = mod.get_function("softmax")
compute_delta2=mod.get_function("compute_delta2")
compute_db=mod.get_function("compute_db")
compute_delta1=mod.get_function("compute_delta1")
update_weights=mod.get_function("update_weights")
update_bias= mod.get_function("update_bias")
TILE_DIM = 16


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
    # print(W1)
    W1 = np.ascontiguousarray(W1, dtype=np.float32)
    # print(W1)
    # Bias of the first layer vector of size d_hidden
    b1 = np.random.rand(1,d_hidden)-0.5 #TODO
    b1 = np.ascontiguousarray(b1, dtype=np.float32)
    # Second layer of size d_hidden x d_output
    W2 = np.random.rand(d_hidden,d_output)-0.5
    W2 = np.ascontiguousarray(W2, dtype=np.float32)
    # The bias of the second layer
    b2 = np.random.rand(1,d_output)-0.5 #TODO
    b2 = np.ascontiguousarray(b2, dtype=np.float32)
    #print(W1)
    W1_gpu = drv.mem_alloc(W1.nbytes)
    b1_gpu = drv.mem_alloc(b1.nbytes)
    W2_gpu = drv.mem_alloc(W2.nbytes)
    b2_gpu = drv.mem_alloc(b2.nbytes)

    # Copier les données sur le GPU
    drv.memcpy_htod(W1_gpu, W1)
    drv.memcpy_htod(b1_gpu, b1)
    drv.memcpy_htod(W2_gpu, W2)
    drv.memcpy_htod(b2_gpu, b2)

    # The model returned at the end is a dictionary of weights and biases
    model = {
        'W1_np' :W1,
        'b1_np' :b1,
        'W2_np' :W2,
        'b2_np' :b2,
        'W1': W1_gpu,
        'b1': b1_gpu,
        'W2': W2_gpu,
        'b2': b2_gpu,
        'W1_shape': W1.shape,
        'b1_shape': b1.shape,
        'W2_shape': W2.shape,
        'b2_shape': b2.shape
    }
    return model


def forward_layer(X_gpu, W_gpu, b_gpu, M, N, K):
    # Allouer de la mémoire pour le résultat sur le GPU
    # W1_temp = np.zeros((N, K), dtype=np.float32)
    # X_temp = np.zeros((M, N), dtype=np.float32)

    # # Copier les données GPU vers l'hôte pour vérification
    # drv.memcpy_dtoh(W1_temp, W_gpu)
    # drv.memcpy_dtoh(X_temp, X_gpu)

    # print("X (input matrix on host):")
    # print(X_temp)
    # print("X wanted:")
    # print(X)
    # print("W (weights matrix on host):")
    # print(W1_temp)
    # print("W wanted:")
    # print(model["W1_np"])
    #print(f"Dimensions: M={M}, N={N}, K={K}")
    C_gpu = drv.mem_alloc(M * K * np.float32().itemsize)

    # Définir la taille des blocs et des grilles pour la multiplication de matrices
    block_size = (TILE_DIM, TILE_DIM, 1)
    grid_size = ((K + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM)

    #print("Debugging info:")
    # print("Dimensions:")
    #print(f"M (rows in X): {M}, N (columns in X, rows in W): {N}, K (columns in W): {K}")
    # print("Block size:", block_size)
    # print("Grid size:", grid_size)

    # Lancer le kernel de multiplication de matrices
    matrix_multiplication(X_gpu, W_gpu, C_gpu, np.int32(M), np.int32(N), np.int32(K), block=block_size, grid=grid_size)

    # Allouer de la mémoire pour le résultat de la multiplication (sur l'hôte)
    C = np.zeros((M, K), dtype=np.float32)
    
    

    # Copier le résultat de C depuis le GPU vers l'hôte
    drv.memcpy_dtoh(C, C_gpu)
    #print("Résultat de la multiplication des matrices (C):")
    #print(C)  # Afficher le résultat de la multiplication des matrices

    #print("Résultat attendu (NumPy):")
    #X_temp = np.array(X_temp, dtype=np.float32)  # Si X est nécessaire sur l'hôte
    #W_temp = np.array(W1_temp, dtype=np.float32)
    expected_C = np.dot(X,model["W1_np"])
    #print(expected_C)

    # Lancer le kernel d'ajout de biais
    add_bias(C_gpu, b_gpu, C_gpu, np.int32(M), np.int32(K), block=block_size, grid=grid_size)

    # Copier les résultats du biais depuis le GPU vers l'hôte
    C_biased = np.zeros((M, K), dtype=np.float32)
    drv.memcpy_dtoh(C_biased, C_gpu)
    #print("Résultat après ajout du biais (C + b):")
    #print(C_biased)  # Afficher le résultat de la multiplication + biais
    #print("Resultat attendue (Numpy) :")
    #print(C+model["b1_np"])

    # Retourner la mémoire du GPU pour C (qui est maintenant C_gpu)
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
def forward_function(model, X):
    # Initialiser les dimensions
    M, N = X.shape
    K1 = model['W1_shape'][1]
    K2 = model['W2_shape'][1]

    # Allouer de la mémoire sur le GPU
    X_gpu = drv.mem_alloc(X.nbytes)

    # Copier les données sur le GPU
    drv.memcpy_htod(X_gpu, X)

    # Première couche
    z1 = forward_layer(X_gpu, W1, b1, M, N, K1)
    a1_gpu = drv.mem_alloc(M*K1*float_size)
    z1_h=np.zeros((M,K1),dtype=np.float32)
    drv.memcpy_dtoh(z1_h,z1)
    #print ("Z :")
    #print(z1_h)
    drv.Context.synchronize()
    sigmoid_activation(z1, a1_gpu, np.int32(M), np.int32(K1), block=(TILE_DIM, TILE_DIM, 1), grid=((K1 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
    a1_h=np.zeros((M,K1),dtype=np.float32)
    drv.memcpy_dtoh(a1_h,a1_gpu)
    #print("sigmoid result :")
    #print(a1_h)
    #print("sigmoid wanted :")
    #print(1/(1+np.exp(-1*z1_h))) 
    drv.Context.synchronize()
    z2 = forward_layer(a1_gpu, W2, b2, M, K1, K2)
    z2_h=np.zeros((M,K2),dtype=np.float32)
    drv.memcpy_dtoh(z2_h,z2)
    #print("Z :")
    #print(z2_h)
    #exp_scores_gpu = drv.mem_alloc(M*K2*float_size)
    #exp_scores(z2, exp_scores_gpu, np.int32(M), np.int32(K2), block=(TILE_DIM, TILE_DIM, 1), grid=((K2 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
    #exp_h=np.zeros((M,K2),dtype=np.float32)
    #drv.memcpy_dtoh(exp_h,exp_scores_gpu)
    #print(exp_h)
    #print("z2 values before softmax:", z2_h)
    probs_gpu = drv.mem_alloc(M*K2*float_size)
    drv.Context.synchronize()
    softmax(z2, probs_gpu, np.int32(M), np.int32(K2), block=(TILE_DIM, TILE_DIM, 1), grid=((K2 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
    ## Necessary to compute the loss on cpu
    probs = np.zeros((M,K2),dtype=np.float32)
        
    drv.memcpy_dtoh(probs, probs_gpu) 
    # print("softmax result :")
    

    return np.argmax(probs,axis=1)


################# TRAINING FUNCTION ###################
def softmax_cpu(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference
def train_model(model, nn_hdim, num_epochs=1, print_loss=False):

    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    float_size=sys.getsizeof(float)

    

    # Allocation d'une matrice temporaire pour copier depuis le GPU
    
    

    # Validation des octets
    
    # Gradient descent. For each batch...
    for i in range(0, num_epochs):
        X_gpu = drv.mem_alloc(X.nbytes)

        # Copie vers le GPU
        drv.memcpy_htod(X_gpu, X)
        y_gpu = drv.mem_alloc(y.nbytes)

        # Copie vers le GPU
        drv.memcpy_htod(y_gpu, y)
        drv.Context.synchronize()
        M, N = X.shape
        K1 = model['W1_shape'][1]
        K2 = model['W2_shape'][1]
        # print("M =",M)
        # print("N =",N)
        # print("K1 =", K1)
        # print("K2 =", K2)
        #print(X)
        
        # W1_temp = np.zeros((N, K1), dtype=np.float32)
       

        # # Copier les données GPU vers l'hôte pour vérification
        # drv.memcpy_dtoh(W1_temp, W1)
        

        
        #print("W (weights matrix on host):")
        #print(W1_temp)
        #print("W wanted:")
        #print(model["W1_np"])
    

        # Forward propagation (copy/paste inside forward_function previously defined)
        z1 = forward_layer(X_gpu, W1, b1, M, N, K1)
        a1_gpu = drv.mem_alloc(M*K1*float_size)
        z1_h=np.zeros((M,K1),dtype=np.float32)
        drv.memcpy_dtoh(z1_h,z1)
        #print ("Z :")
        #print(z1_h)
        drv.Context.synchronize()
        sigmoid_activation(z1, a1_gpu, np.int32(M), np.int32(K1), block=(TILE_DIM, TILE_DIM, 1), grid=((K1 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
        a1_h=np.zeros((M,K1),dtype=np.float32)
        drv.memcpy_dtoh(a1_h,a1_gpu)
        #print("sigmoid result :")
        #print(a1_h)
        #print("sigmoid wanted :")
        #print(1/(1+np.exp(-1*z1_h))) 
        drv.Context.synchronize()
        z2 = forward_layer(a1_gpu, W2, b2, M, K1, K2)
        z2_h=np.zeros((M,K2),dtype=np.float32)
        drv.memcpy_dtoh(z2_h,z2)
        #print("Z :")
        #print(z2_h)
        #exp_scores_gpu = drv.mem_alloc(M*K2*float_size)
        #exp_scores(z2, exp_scores_gpu, np.int32(M), np.int32(K2), block=(TILE_DIM, TILE_DIM, 1), grid=((K2 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
        #exp_h=np.zeros((M,K2),dtype=np.float32)
        #drv.memcpy_dtoh(exp_h,exp_scores_gpu)
        #print(exp_h)
        #print("z2 values before softmax:", z2_h)
        probs_gpu = drv.mem_alloc(M*K2*float_size)
        drv.Context.synchronize()
        softmax(z2, probs_gpu, np.int32(M), np.int32(K2), block=(TILE_DIM, TILE_DIM, 1), grid=((K2 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
        ## Necessary to compute the loss on cpu
        probs = np.zeros((M,K2),dtype=np.float32)
         
        drv.memcpy_dtoh(probs, probs_gpu) 
        # print("softmax result :")
        # print(probs) 
        # print("softmax wanted : ")
        # print(softmax_cpu(z2_h))    
        
        #print (probs[np.arange(probs.shape[0]), y])
        
        correct_logprobs = np.log(probs[np.arange(probs.shape[0]), y])# Calculation of cross entropy for each example
        
        data_loss = -1./N * np.sum(correct_logprobs,axis=0,keepdims=True) # Loss totale
        
        
        
        # Backpropagation
        #TODO
        #Computing delta2 as the difference between the output of the model and the ground truth
       # delta2 calculation
        # print("Probs :",probs.nbytes)
        # print("Y :", )
        grid_size = (probs.shape[0] + TILE_DIM - 1) // TILE_DIM
        delta2_gpu = drv.mem_alloc(probs.nbytes)
        drv.Context.synchronize()
        compute_delta2(probs_gpu, y_gpu, delta2_gpu, np.int32(probs.shape[1]), np.int32(probs.shape[0]),
                    block=(probs.shape[0], 1, 1),
                    grid=(1,1))
        # delta2_cpu=probs.copy()
        # delta2_cpu[np.arange(probs.shape[0]), y_temp] -= 1
        delta2_h = np.zeros((M,K2),dtype=np.float32)
        drv.memcpy_dtoh(delta2_h, delta2_gpu) 
        # y_cpu=np.zeros(probs.shape[0],dtype=np.int32)
        # drv.memcpy_dtoh(y_cpu, y_gpu) 
        #print(" delta 2 results : ")
        #print(delta2_h)
        # print("delta2 wanted : ")
        # print(delta2_cpu)
        # print("differences :")
        # print(delta2_cpu-delta2_h)
        # print("diif of y :")
        # print(y_cpu-y)
        # Transpose a1
        grid_y = (K1 + TILE_DIM - 1) // TILE_DIM
        grid_x = (M + TILE_DIM - 1) // TILE_DIM
        a1_gpu_T = drv.mem_alloc(M * K1 * float_size)
        transpose(a1_gpu, a1_gpu_T, np.int32(M), np.int32(K1),
                block=(TILE_DIM, TILE_DIM, 1),
                grid=(grid_x,grid_y))
        # a1_h_T=np.zeros((K1,M),dtype=np.float32)
        # drv.memcpy_dtoh(a1_h_T,a1_gpu_T)
        # print("CPU :")
        # print(a1_h.T)
        # print("GPU : ")
        # print(a1_h_T)
        # # Gradient dW2
        dW2_gpu = drv.mem_alloc(K1 * K2 * float_size)
        
        matrix_multiplication(a1_gpu_T, delta2_gpu, dW2_gpu, np.int32(K1), np.int32(M), np.int32(K2),
                            block=(TILE_DIM, TILE_DIM, 1),
                            grid=((K2 + TILE_DIM - 1) // TILE_DIM, (K1 + TILE_DIM - 1) // TILE_DIM))
        # dW2_h=np.zeros((K1,K2),dtype=np.float32)
        # drv.memcpy_dtoh(dW2_h, dW2_gpu) 
        # dW2_cpu=np.dot(a1_h.T,delta2_h)

        # print("Diff DW2 :")
        # print(dW2_h-dW2_cpu)
        # print("GPU : ")
        # print(dW2_h)
        # print("CPU : ")
        # print(dW2_cpu)
        # Compute db2
        db2_gpu = drv.mem_alloc(K2 * float_size)
        compute_db(delta2_gpu, db2_gpu, np.int32(K2), np.int32(M),
                block=(K2, 1, 1),
                grid=(1, 1))
        # db2_h=np.zeros(K2,dtype=np.float32)
        # drv.memcpy_dtoh(db2_h,db2_gpu)
        # print(delta2_h)
        # print("GPU :")
        # print(db2_h)
        # print("CPU :")
        # print(np.sum(delta2_h, axis=0, keepdims=True) )
        # print("Diff :")
        # print(db2_h-np.sum(delta2_h, axis=0, keepdims=True))
        # delta1 calculation
        
        W2_T = drv.mem_alloc(K1 * K2 * float_size)
        grid_y = (K2 + TILE_DIM - 1) // TILE_DIM
        grid_x = (K1 + TILE_DIM - 1) // TILE_DIM
        transpose(W2, W2_T, np.int32(K1), np.int32(K2),
                block=(TILE_DIM, TILE_DIM, 1),
                grid=(grid_x,grid_y))
        # w2_trans=np.zeros((K2,K1),dtype=np.float32)
        # drv.memcpy_dtoh(w2_trans,W2_T)
        # print("Trans W2 :")
        # print(w2_trans.shape)

        # print("Wanted trans :")
        # print(model["W2_np"].T)
        delta_dot_gpu = drv.mem_alloc(M * K1 * float_size)
        matrix_multiplication(delta2_gpu, W2_T, delta_dot_gpu, np.int32(M), np.int32(K2), np.int32(K1),
                            block=(TILE_DIM, TILE_DIM, 1),
                            grid=((K1 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
        prod=np.zeros((M,K1),dtype=np.float32)
        drv.memcpy_dtoh(prod,delta_dot_gpu)
      
        # print("Prod :")
        # print(prod)
        # print("Wanted prod : ")
        # print(np.dot(delta2_h,model["W2_np"].T))
        # print(prod.shape)
        # print(z1_h.shape)
        delta1_gpu = drv.mem_alloc(M * K1 * float_size)
        compute_delta1( delta_dot_gpu,z1, delta1_gpu, np.int32(M), np.int32(K1),
                    block=(TILE_DIM, TILE_DIM, 1),
                    grid=((M + TILE_DIM - 1) // TILE_DIM, (K1 + TILE_DIM - 1) // TILE_DIM))
        delta1_h=np.zeros((M,K1),dtype=np.float32)   
        drv.memcpy_dtoh(delta1_h,delta1_gpu)
        # print("GPU : ")
        # print(delta1_h)
        # print("CPU :")
        # print((sigmoid_derivative(z1_h)*np.dot(delta2_h,model["W2_np"].T)).shape)
        # print("Diff : ")
        # print(delta1_h-sigmoid_derivative(z1_h)*np.dot(delta2_h,model["W2_np"].T))
        dW1_gpu = drv.mem_alloc(N*K1*float_size)
        
        db1_gpu = drv.mem_alloc(K1*float_size)
        X_gpu_T=drv.mem_alloc(X.nbytes)
        grid_y = (N + TILE_DIM - 1) // TILE_DIM
        grid_x = (M + TILE_DIM - 1) // TILE_DIM
        transpose(X_gpu,X_gpu_T,np.int32(M),np.int32(N),block=(TILE_DIM, TILE_DIM, 1), grid=(grid_x,grid_y))
        X_T_h=np.zeros((N,M),dtype=np.float32)
        drv.memcpy_dtoh(X_T_h,X_gpu_T)
        # print("Trans :")
        # print(X_T_h)
        # print("Trans wanted :")
        # print(X.T)
        matrix_multiplication(X_gpu_T, delta1_gpu, dW1_gpu, np.int32(N), np.int32(M), np.int32(K1),
                                 block=(TILE_DIM, TILE_DIM, 1),
                                  grid=((K1 + TILE_DIM - 1) // TILE_DIM, (N + TILE_DIM - 1) // TILE_DIM))
        compute_db(delta1_gpu, db1_gpu, np.int32(K1), np.int32(M),
                    block=(K1, 1, 1), 
                    grid=(1, 1))  # Gradient of the biais of the hidden layer
        dW1_cpu=np.zeros_like(model["W1_np"])
        drv.memcpy_dtoh(dW1_cpu,dW1_gpu)
        dW1_h=np.dot(X.T,delta1_h)
        # print("Diff entre dw1:")
        # print(dW1_cpu-dW1_h)
        
        # Gradient descente
        new_W1=drv.mem_alloc(N*K1*float_size)
        new_W2=drv.mem_alloc(K1*K2*float_size)
        new_b1=drv.mem_alloc(K1*float_size)
        new_b2=drv.mem_alloc(K2*float_size)

        
        update_weights(W1, dW1_gpu, new_W1, np.float32(epsilon), np.int32(K1), np.int32(M),
                        block=(TILE_DIM, 1, 1), 
                        grid=((M + TILE_DIM - 1) // TILE_DIM, 1))

        new_W1_h=np.zeros_like(model["W1_np"])
        drv.memcpy_dtoh(new_W1_h,new_W1)
        print("Result update_weights :")
        print(new_W1_h)
        print("Diff with previous")
        print(new_W1_h-model["W1_np"])
        new_W1_cpu=model["W1_np"]-epsilon*dW1_cpu
        print("Diff with expected :")
        print(new_W1_cpu-new_W1_h)


        update_bias(b1, db1_gpu, new_b1, np.float32(epsilon), np.int32(K1), 
                    block=(TILE_DIM, 1, 1),
                    grid=((K1 + TILE_DIM - 1) // TILE_DIM, 1))
        update_weights(W2, dW2_gpu, new_W2, np.float32(epsilon), np.int32(K2), np.int32(K1), 
                        block=(TILE_DIM, 1, 1), 
                        grid=((K1 + TILE_DIM - 1) // TILE_DIM,1))
        update_bias(b2, db2_gpu, new_b2, np.float32(epsilon), np.int32(K2), 
                    block=(TILE_DIM, 1, 1), 
                    grid=((M + TILE_DIM - 1) // TILE_DIM, 1))
        
        #Updating weights and biases

        W1 = new_W1
        W2= new_W2
        b1=new_b1        
        b2=new_b2 
        a1_gpu.free()
        z1.free()
        z2.free()
        probs_gpu.free()
        delta2_gpu.free()
        delta1_gpu.free()
        dW2_gpu.free()
        dW1_gpu.free()
        db2_gpu.free()
        db1_gpu.free()
        a1_gpu_T.free()
        X_gpu_T.free()
        X_gpu.free()
        y_gpu.free()
        W2_T.free()
        # Loss display
        if print_loss and i % 1 == 0:
          print("Loss at epoch %i: %f" %(i, data_loss))
    

    return model
######################## PREDICTION FUNCTION ###############
def predict(model, x):
    M, N = x.shape
    float_size=sys.getsizeof(float)
    K1 = model['W1_shape'][1]
    K2 = model['W2_shape'][1]
    W1 = model['W1']
    b1 = model['b1']
    W2 = model['W2']
    b2 = model['b2']
    # Allouer de la mémoire sur le GPU
    X_gpu = drv.mem_alloc(x.nbytes)

    # Copier les données sur le GPU
    drv.memcpy_htod(X_gpu, x)

    # Première couche
    z1 = forward_layer(X_gpu, W1, b1, M, N, K1)
    a1_gpu = drv.mem_alloc(M*K1*float_size)
    z1_h=np.zeros((M,K1),dtype=np.float32)
    drv.memcpy_dtoh(z1_h,z1)
    #print ("Z :")
    #print(z1_h)
    drv.Context.synchronize()
    sigmoid_activation(z1, a1_gpu, np.int32(M), np.int32(K1), block=(TILE_DIM, TILE_DIM, 1), grid=((K1 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
    a1_h=np.zeros((M,K1),dtype=np.float32)
    drv.memcpy_dtoh(a1_h,a1_gpu)
    #print("sigmoid result :")
    #print(a1_h)
    #print("sigmoid wanted :")
    #print(1/(1+np.exp(-1*z1_h))) 
    drv.Context.synchronize()
    z2 = forward_layer(a1_gpu, W2, b2, M, K1, K2)
    z2_h=np.zeros((M,K2),dtype=np.float32)
    drv.memcpy_dtoh(z2_h,z2)
    #print("Z :")
    #print(z2_h)
    #exp_scores_gpu = drv.mem_alloc(M*K2*float_size)
    #exp_scores(z2, exp_scores_gpu, np.int32(M), np.int32(K2), block=(TILE_DIM, TILE_DIM, 1), grid=((K2 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
    #exp_h=np.zeros((M,K2),dtype=np.float32)
    #drv.memcpy_dtoh(exp_h,exp_scores_gpu)
    #print(exp_h)
    #print("z2 values before softmax:", z2_h)
    probs_gpu = drv.mem_alloc(M*K2*float_size)
    drv.Context.synchronize()
    softmax(z2, probs_gpu, np.int32(M), np.int32(K2), block=(TILE_DIM, TILE_DIM, 1), grid=((K2 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
    ## Necessary to compute the loss on cpu
    probs = np.zeros((M,K2),dtype=np.float32)
        
    drv.memcpy_dtoh(probs, probs_gpu) 
    # print("softmax result :")
    

    return np.argmax(probs,axis=1)

########################## TEST ########################## number of examples in the training set

np.random.seed(1)
X, y = sklearn.datasets.make_moons(300, noise=0.2)
X = np.ascontiguousarray(X, dtype=np.float32)
#y_temp=y.copy()
y= np.ascontiguousarray(y, dtype=np.int32)
#y = np.ascontiguousarray(y, dtype=np.float32)
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
model = train_model(model,d_hidden, num_epochs=1, print_loss=True)

print("The final accuracy obtained is :", accuracy(y, predict(model, X)))
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
plot_decision_boundary(lambda v: predict(model,v))
plt.title("Decision Boundary for hidden layer size 3")
plt.show()