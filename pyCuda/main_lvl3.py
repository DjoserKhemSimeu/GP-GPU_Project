
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
compute_dW=mod.get_function("compute_dW")
compute_delta1=mod.get_function("compute_delta1")
update_weights=mod.get_function("update_weights")
update_bias= mod.get_function("update_bias")
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
class Model :
    def __init__(self, nb_hidden, param):
        self.nb_hidden = nb_hidden
        self.param = param

def init_model(d_input: int, d_hidden: list, d_output: int):
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
    weights_np = []
    weights_gpu = []
    biases_np = []
    biases_gpu =[]
    W_shapes=[]
    b_shapes=[]
    # First layer of size d_input x d_hidden
    W1 = np.random.rand(d_input,d_hidden[0])-0.5 #TODO
    # print(W1)
    W1 = np.ascontiguousarray(W1, dtype=np.float32)

    b1 = np.random.rand(1,d_hidden[0])-0.5 #TODO
    b1 = np.ascontiguousarray(b1, dtype=np.float32)
    
    W1_gpu = drv.mem_alloc(W1.nbytes)
    b1_gpu = drv.mem_alloc(b1.nbytes)
    drv.memcpy_htod(W1_gpu, W1)
    drv.memcpy_htod(b1_gpu, b1)
    weights_np.append(W1)
    biases_np.append(b1)
    weights_gpu.append(W1_gpu)
    biases_gpu.append(b1_gpu)
    W_shapes.append(W1.shape)
    b_shapes.append(b1.shape)

    for i in range(1, len(d_hidden)):
        Wi = np.random.rand(d_hidden[i-1], d_hidden[i])-0.5 #TODO
   
        Wi = np.ascontiguousarray(Wi, dtype=np.float32)

        bi = np.random.rand(1,d_hidden[i])-0.5 #TODO
        bi = np.ascontiguousarray(bi, dtype=np.float32)
        
        Wi_gpu = drv.mem_alloc(Wi.nbytes)
        bi_gpu = drv.mem_alloc(bi.nbytes)
        drv.memcpy_htod(Wi_gpu, Wi)
        drv.memcpy_htod(bi_gpu, bi)
        weights_np.append(Wi)
        biases_np.append(bi)
        weights_gpu.append(Wi_gpu)
        biases_gpu.append(bi_gpu)
        W_shapes.append(Wi.shape)
        b_shapes.append(bi.shape)



    # Bias of the first layer vector of size d_hidden
    
    # Second layer of size d_hidden x d_output
    W_out = np.random.rand(d_hidden[-1],d_output)-0.5
    W_out = np.ascontiguousarray(W_out, dtype=np.float32)
    # The bias of the second layer
    b_out = np.random.rand(1,d_output)-0.5 #TODO
    b_out = np.ascontiguousarray(b_out, dtype=np.float32)
    #print(W1)
    
    W_out_gpu = drv.mem_alloc(W_out.nbytes)
    b_out_gpu = drv.mem_alloc(b_out.nbytes)

    # Copier les données sur le GPU
    
    drv.memcpy_htod(W_out_gpu, W_out)
    drv.memcpy_htod(b_out_gpu, b_out)
    weights_np.append(W_out)
    biases_np.append(b_out)
    weights_gpu.append(W_out_gpu)
    biases_gpu.append(b_out_gpu)
    W_shapes.append(W_out.shape)
    b_shapes.append(b_out.shape)

    # The model returned at the end is a dictionary of weights and biases
    model_params = {
        'W_np': weights_np,
        'W_gpu': weights_gpu,
        'b_np': biases_np,
        'b_gpu': biases_gpu,
        'W_shapes': W_shapes,
        'b_shapes': b_shapes
    }

    model = Model(nb_hidden=len(d_hidden), param=model_params)
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
    # C = np.zeros((M, K), dtype=np.float32)
    
    

    # Copier le résultat de C depuis le GPU vers l'hôte
    # drv.memcpy_dtoh(C, C_gpu)
    #print("Résultat de la multiplication des matrices (C):")
    #print(C)  # Afficher le résultat de la multiplication des matrices

    #print("Résultat attendu (NumPy):")
    #X_temp = np.array(X_temp, dtype=np.float32)  # Si X est nécessaire sur l'hôte
    #W_temp = np.array(W1_temp, dtype=np.float32)
    #expected_C = np.dot(X,model["W1_np"])
    #print(expected_C)

    # Lancer le kernel d'ajout de biais
    add_bias(C_gpu, b_gpu, C_gpu, np.int32(M), np.int32(K), block=block_size, grid=grid_size)

    # Copier les résultats du biais depuis le GPU vers l'hôte
    # C_biased = np.zeros((M, K), dtype=np.float32)
    # drv.memcpy_dtoh(C_biased, C_gpu)
    # #print("Résultat après ajout du biais (C + b):")
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
    # z1_h=np.zeros((M,K1),dtype=np.float32)
    # drv.memcpy_dtoh(z1_h,z1)
    #print ("Z :")
    #print(z1_h)
    drv.Context.synchronize()
    sigmoid_activation(z1, a1_gpu, np.int32(M), np.int32(K1), block=(TILE_DIM, TILE_DIM, 1), grid=((K1 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
    # a1_h=np.zeros((M,K1),dtype=np.float32)
    # drv.memcpy_dtoh(a1_h,a1_gpu)
    #print("sigmoid result :")
    #print(a1_h)
    #print("sigmoid wanted :")
    #print(1/(1+np.exp(-1*z1_h))) 
    drv.Context.synchronize()
    z2 = forward_layer(a1_gpu, W2, b2, M, K1, K2)
    # z2_h=np.zeros((M,K2),dtype=np.float32)
    # drv.memcpy_dtoh(z2_h,z2)
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
    exp_scores = np.exp(x)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
def train_model(model, nn_hdim, num_epochs=1,epsilon=0.001, print_loss=False):

    params = model.param
    float_size=sys.getsizeof(float)

    
    
    # Allocation d'une matrice temporaire pour copier depuis le GPU
    
    

    # Validation des octets
    start_loop=time.time()
    # Gradient descent. For each batch...
    for i in range(0, num_epochs):
        #print(f"###################### EPOCH {i} ########################")
        start=time.time()
        X_gpu = drv.mem_alloc(X.nbytes)

        # Copie vers le GPU
        drv.memcpy_htod(X_gpu, X)
        y_gpu = drv.mem_alloc(y.nbytes) #############################################

        # Copie vers le GPU
        drv.memcpy_htod(y_gpu, y)
        
        M, N = X.shape
        
        K_out=params['W_shapes'][-1][1]
  
        start_fw=time.time()

        activations = [X_gpu]
        activations_cpu=[X]
        zs = []
        # print(" FORWARD PASS ::")
        for j, (w, b) in enumerate(zip(params["W_gpu"], params["b_gpu"])):
            # print(f"#### Layer {j} ####")
            
            K0=params['W_shapes'][j][0]
            K1=params['W_shapes'][j][1]

            # print(f"K0 : {K0}, K1 : {K1}")
            z = forward_layer(activations[-1], w, b, M,K0,K1)
            # act_h=np.zeros((M,K0),dtype=np.float32)
            
            # b_h=np.zeros_like(params["b_np"][j])
            # drv.memcpy_dtoh(b_h,b)
            # drv.memcpy_dtoh(act_h,activations[-1])
            
            # w_h=np.zeros_like(params["W_np"][j])
            # drv.memcpy_dtoh(w_h,w)
            # wanted= np.dot(act_h,w_h)+b_h
            # print("Weights : ", w_h)
            # print("Activations : ",act_h)
            # print("Wanted : ",wanted)
            
            
        
            z_h=np.zeros((M,K1),dtype=np.float32)
            drv.memcpy_dtoh(z_h,z)
            # print("Obtained : ",z_h)
            # print("Diff : " , z_h-wanted)
            
            

            zs.append(z)
            act_cpu=np.zeros((M,K1),dtype=np.float32)
            act_gpu=drv.mem_alloc(M*K1*float_size)
            if j < len(params["W_gpu"]) - 1:  # Apply the sigmoid on the hidden layer
                sigmoid_activation(z, act_gpu, np.int32(M), np.int32(K1),
                                     block=(TILE_DIM, TILE_DIM, 1), 
                                     grid=((K1 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
            else:
                softmax(z, act_gpu, np.int32(M), np.int32(K1),
                        block=(TILE_DIM, TILE_DIM, 1), 
                        grid=((K1 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
                want_soft=softmax_cpu(z_h)
            activations.append(act_gpu)
            # drv.memcpy_dtoh(act_cpu,act_gpu)
            # activations_cpu.append(act_cpu)
        
        probs_gpu=activations[-1]
        # Forward propagation (copy/paste inside forward_function previously defined)
        
       
        probs = np.zeros((M,K_out),dtype=np.float32)
         
        drv.memcpy_dtoh(probs, probs_gpu) 
        #print(probs)
        
        # if i==0:
        #    print("Time to compute the forward pass =", end_fw-start_fw)
        
        
        
        correct_logprobs = np.log(probs[np.arange(probs.shape[0]), y])# Calculation of cross entropy for each example
        
        data_loss = -1./N * np.sum(correct_logprobs,axis=0,keepdims=True) # Loss totale
        
        
        
        # Backpropagation
        #TODO
        #Computing delta2 as the difference between the output of the model and the ground truth
       # delta2 calculation
        # print("Probs :",probs.nbytes)
        # print("Y :", )
        # print ("DELTA COMPUTATION ::")
        
        
        grid_size = (probs.shape[0] + TILE_DIM - 1) // TILE_DIM
        delta2_gpu = drv.mem_alloc(probs.nbytes)
        
        compute_delta2(probs_gpu, y_gpu, delta2_gpu, np.int32(probs.shape[1]), np.int32(probs.shape[0]),
                    block=(TILE_DIM, 1, 1),
                    grid=(grid_size,1))
        deltas=[delta2_gpu]
        # d2_h=np.zeros_like(probs)
        # drv.memcpy_dtoh(d2_h,delta2_gpu)
        # d2_cpu=[probs]
        # d2_cpu[-1][range(len(y)), y] -= 1
        # print("Obtained : ",d2_h)
        # print("Wanted : ",d2_cpu[0])
        # print("Diff : ",d2_h-d2_cpu[0])
        
        
        for j in range(len(params["W_gpu"])-2, -1, -1):
            # print(f"##### {j}")

            K0=params['W_shapes'][j][0]
            K1=params['W_shapes'][j+1][0]
            K2=params['W_shapes'][j+1][1]
            
            Wi=params["W_gpu"][j]
            Wii=params["W_gpu"][j+1]
            # Wii_h=np.zeros_like(params['W_np'][j+1])
            # drv.memcpy_dtoh(Wii_h,Wii)
            # print(f"K0 : {K0}, K1 : {K1}, k2 : {K2}")
            Wii_T = drv.mem_alloc(params['W_np'][j+1].nbytes)
            grid_y = (K2 + TILE_DIM - 1) // TILE_DIM
            grid_x = (K1 + TILE_DIM - 1) // TILE_DIM
            transpose(Wii, Wii_T, np.int32(K1), np.int32(K2),
                    block=(TILE_DIM, TILE_DIM, 1),
                    grid=(grid_x,grid_y))
            # Wii_T_h=np.zeros((K2,K1),dtype=np.float32)
            # drv.memcpy_dtoh(Wii_T_h,Wii_T)
            # print("Wanted : ",Wii_h)
            # print("Obtained : ",Wii_T_h)
            # print("Diff transpose : ",Wii_T_h-Wii_h.T)
            delta_dot_gpu = drv.mem_alloc(M * K1 * float_size)
            matrix_multiplication(deltas[len(params["W_gpu"])-2-j], Wii_T, delta_dot_gpu, np.int32(M), np.int32(K2), np.int32(K1),
                            block=(TILE_DIM, TILE_DIM, 1),
                            grid=((K1 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
            # delta_dot_h=np.zeros((M,K1),dtype=np.float32)
            # drv.memcpy_dtoh(delta_dot_h,delta_dot_gpu)
            # delta_h=np.zeros((M,K2),dtype=np.float32)
            # drv.memcpy_dtoh(delta_h,deltas[len(params["W_gpu"])-2-j])
            # print("Delta to use :", delta_h)
            # print("Wanted delta dot : ",np.dot(delta_h,Wii_T_h))
            # print("Obtained delta : ",delta_dot_h)
            # print( "Diff : ",delta_dot_h-np.dot(delta_h,Wii_T_h))
            delta_gpu = drv.mem_alloc(M * K1 * float_size)
            # z_h = np.zeros((M,K1),dtype=np.float32)
            # drv.memcpy_dtoh(z_h,zs[j])
            # d1_cpu=sigmoid_derivative(z_h) *delta_dot_h
            compute_delta1( delta_dot_gpu,zs[j], delta_gpu, np.int32(M), np.int32(K1),
                        block=(TILE_DIM, TILE_DIM, 1),
                        grid=((M + TILE_DIM - 1) // TILE_DIM, (K1 + TILE_DIM - 1) // TILE_DIM))
            # d1_h=np.zeros((M,K1),dtype=np.float32)
            # drv.memcpy_dtoh(d1_h,delta_gpu)
            # print("Wanted : ",d1_cpu)
            # print("Obtained : ",d1_h)
            # print("Diff : ",d1_h-d1_cpu)
            deltas.append(delta_gpu)
            
            

        deltas.reverse()

        # print(" UPDATES PARAMS :")
        for j in range(len(params["W_gpu"])):
            # print(f"#### Layer {j} ####")
            K0=params['W_shapes'][j][0]
            K1=params['W_shapes'][j][1]
            # print(f"K0 : {K0}, K1 : {K1}")
            # act_h=np.zeros((M,K0),dtype=np.float32)
            # drv.memcpy_dtoh(act_h,activations[j])
            # print("Initial : ", act_h)
            act_gpu_T = drv.mem_alloc(M * K0 * float_size)
            grid_y = (K0 + TILE_DIM - 1) // TILE_DIM
            grid_x = (M + TILE_DIM - 1) // TILE_DIM
            transpose(activations[j], act_gpu_T, np.int32(M), np.int32(K0),
                    block=(TILE_DIM, TILE_DIM, 1),
                    grid=(grid_x,grid_y))
            # act_T_h=np.zeros((K0,M),dtype=np.float32)
            # drv.memcpy_dtoh(act_T_h,act_gpu_T)
            # print ("After trans : ", act_T_h)
            # print("Diff : ",act_T_h-act_h.T )
            dW_gpu = drv.mem_alloc(K0 * K1 * float_size)
        
            compute_dW(act_gpu_T, deltas[j], dW_gpu, np.int32(K0), np.int32(M), np.int32(K1), params['W_gpu'][j], np.float32(epsilon),
                            block=(TILE_DIM, TILE_DIM, 1),
                            grid=((K1 + TILE_DIM - 1) // TILE_DIM, (K0 + TILE_DIM - 1) // TILE_DIM))
            params["W_gpu"][j]=dW_gpu
            db_gpu = drv.mem_alloc(K1 * float_size)
            compute_db(deltas[j], db_gpu, np.int32(K1), np.int32(M),params["b_gpu"][j],np.float32(epsilon),
                    block=(TILE_DIM, 1, 1),
                    grid=((K1 + TILE_DIM - 1) // TILE_DIM, 1))
            params["b_gpu"][j]=db_gpu
            # dW_h= np.zeros((K0,K1),dtype=np.float32)
            # drv.memcpy_dtoh(dW_h,dW_gpu)
            # print("new W : ",dW_h)
        # for d in deltas:
        #      d.free()
        # for a in activations:
        #      a.free()
        # for z in zs:
        #      z.free()
        end= time.time()
        elapsed_time=end-start
        #print("elapsed:", elapsed_time)
        if i==0:
            print("elapsed time for 1 epoch:", elapsed_time)
            with open(f'../data/epoch_timeslvl3_{TILE_DIM}.txt', 'a') as file:
                file.write(f"{nn_hdim[1]} {elapsed_time}\n")
        if print_loss and i % 1 == 0:
          print("Loss at epoch %i: %f" %(i, data_loss))
    end_loop=time.time()
    loop_time=end_loop-start_loop
    print("Looping time : ", end_loop-start_loop)
    with open(f'../data/loop_timeslvl3_{TILE_DIM}.txt', 'a') as file:
        file.write(f"{nn_hdim[1]} {loop_time}\n")
    for i in range(len(params['W_gpu'])):
        W_h=np.zeros_like(params['W_np'][i])
        drv.memcpy_dtoh(W_h,params['W_gpu'][i])
        # print("diff W : ", W_h-params['W_np'][i])
        params['W_np'][i]=W_h
        b_h=np.zeros_like(params['b_np'][i])
        drv.memcpy_dtoh(b_h,params['b_gpu'][i])
        #print("diff b : ", b_h-params['b_np'][i])
        params['b_np'][i]=b_h

    return model
######################## PREDICTION FUNCTION ###############
def predict(model, x):
    float_size=sys.getsizeof(float)
    params=model.param
    X_gpu = drv.mem_alloc(X.nbytes)

    # Copie vers le GPU
    drv.memcpy_htod(X_gpu, x)


    
    M, N = X.shape
    
    K_out=params['W_shapes'][-1][1]

  

    activations = [X_gpu]
    zs = []
    # print(" FORWARD PASS ::")
    for j, (w, b) in enumerate(zip(params["W_gpu"], params["b_gpu"])):
        # print(f"#### Layer {j} ####")
        K0=params['W_shapes'][j][0]
        K1=params['W_shapes'][j][1]
        z = forward_layer(activations[-1], w, b, M,K0,K1)
        
        
        

        zs.append(z)
        act_gpu=drv.mem_alloc(M*K1*float_size)
        if j < len(params["W_gpu"]) - 1:  # Apply the sigmoid on the hidden layer
            sigmoid_activation(z, act_gpu, np.int32(M), np.int32(K1),
                                    block=(TILE_DIM, TILE_DIM, 1), 
                                    grid=((K1 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
        else:
            softmax(z, act_gpu, np.int32(M), np.int32(K1),
                    block=(TILE_DIM, TILE_DIM, 1), 
                    grid=((K1 + TILE_DIM - 1) // TILE_DIM, (M + TILE_DIM - 1) // TILE_DIM))
        activations.append(act_gpu)
        
    probs_gpu=activations[-1]
    # Forward propagation (copy/paste inside forward_function previously defined)
    
    
    probs = np.zeros((M,K_out),dtype=np.float32)
        
    drv.memcpy_dtoh(probs, probs_gpu) 
# print(probs)
    return np.argmax(probs,axis=1)

########################## TEST ########################## number of examples in the training set
def forward_layer_cpu(X, W, b):
    v= np.dot(X,W)
    return v+b
def predict_cpu(model, x):
    float_size=sys.getsizeof(float)
    params=model.param
    # X_gpu = drv.mem_alloc(x.nbytes)
    # M, N = x.shape
    # K_out=params['W_shapes'][-1][1]
    activations = [x]
    for j, (w, b) in enumerate(zip(params["W_np"], params["b_np"])):
        #print(f"#### Layer {j} ####")
        K0=params['W_shapes'][j][0]
        K1=params['W_shapes'][j][1]
        z = forward_layer_cpu(activations[-1], w, b)
        
        

        
        
        if j < len(params["W_gpu"]) - 1:  # Apply the sigmoid on the hidden layer
            act=sigmoid(z)
        else:
            act=softmax_cpu(z)
        activations.append(act)
        
    probs=activations[-1]
    
    
    return np.argmax(probs,axis=1)
np.random.seed(1)
X, y = sklearn.datasets.make_moons(600, noise=0.01)
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
#d_hidden = 2 #TODO 

d_input = 2
d_hidden = [32,32,32,32] # 4 hidden layers with 32 neurons for each ones
d_output = 2
with open(f'../data/epoch_timeslvl3_{TILE_DIM}.txt', 'w') as file:
    pass
with open(f'../data/loop_timeslvl3_{TILE_DIM}.txt', 'w') as file:
    pass
for h in [2,4,8,16,32,64,128,254,512,1024]:
    d_hidden = [h]*4
    model = init_model(d_input, d_hidden, d_output)

    start= time.time()
    model = train_model(model,d_hidden, num_epochs=1000, print_loss=False)
    end=time.time()
    print(f"N_Hidden : {d_hidden[0]}   Training time : {end-start}")


    print("The final accuracy obtained is :", accuracy(y, predict_cpu(model, X)))
    for j in range(len(model.param["W_gpu"])):
        model.param["W_gpu"][j].free()
        model.param["b_gpu"][j].free()

def plot_perf():
    # Lire les données des fichiers texte
    with open('../data/loop_timeslvl00.txt', 'r') as file:
        loop_lvl0 = file.readlines()
    with open('../data/loop_timeslvl3.txt', 'r') as file:
        loop_lvl2 = file.readlines()
    with open('../data/epoch_timeslvl00.txt', 'r') as file:
        epoch_lvl0 = file.readlines()
    with open('../data/epoch_timeslvl3.txt', 'r') as file:
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
plt.title("Decision Boundary for hidden layer size 3")
plt.show()
