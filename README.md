# GP-GPU Computing project
### Djoser SIMEU M2 MOSIG 
In the context of the GP-GPU computing project, I have as project to parallelize at GPU levels the computation of a MLP model for the trainning and the inference process.
# Architecture of the project 
The project is in python programming language with the usage of the Numba python library a NVIDIA Gpu Computing framework. I divided this project into two main part : The paralellization of an MLP model with one hidden layer, and the paralellization of an MLP model with $n$ hidden layer where $n$ is a parameters of the model. For both paralellization problems, I defined multiple parallelization levels where the levels 0 correspond to a sequential execution on CPU, their are modelized by the files ```main_lvlN.py```.
# First Step : Matrix Computation parallelization
