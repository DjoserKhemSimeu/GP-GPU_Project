# GP-GPU Computing project
### Djoser SIMEU M2 MOSIG 
In the context of the GP-GPU computing project, I have as project to parallelize at GPU levels the computation of a MLP model for the trainning and the inference process.
# Architecture of the project 
The project is in python programming language with the usage of the pyCuda python library a NVIDIA Gpu Computing framework. I divided this project into two main part : The paralellization of an MLP model with one hidden layer, and the paralellization of an MLP model with $n$ hidden layer where $n$ is a parameters of the model. For both paralellization problems, I defined multiple parallelization levels where the levels 0 correspond to a sequential execution on CPU, their are modelized by the files ```main_lvlN.py```.
# First Step : Matrix Computation parallelization
The first objective is to only parallelize the matrix operation made on the program: 

``` python
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

```

