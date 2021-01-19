import numpy
import numpy as np
import math
from math import exp
#from .layers import Activation_Function


'''Activation Functions:
       1. Linear or identity
       2. Sinusoid
       3. Tanh
       4. Sigmoid, Logistic, or softstep
       5. Relu
       6. Leaky Relu
       7. Hard Tanh
                                          '''

class Activation_Function:

    def __init__(self, *args, **kwargs):
        self.cache = {}
        self.grad = {}

    def __call__(self, *args, **kwargs):
        output = self.forward(*args, **kwargs)
        self.grad = self.local_grad(*args, **kwargs)

    def forward(self, *args):
        'Returns the Output of the activation function of the input'

        pass

    def backward (self, *args, **kwargs):
        'Returns the differentiation of the activation function of the input after the Forward path'
        pass
   
    def local_grad(self, *args):
        'Returns the differentiation of the activation function of the input'
        pass


#Activation Function of linear or identity
class Linear(Activation_Function):
    def forward(self, x):
        return x

    def backward(self, dY):
        return dY * self.grad['x']


    def local_grad(self, x):
        y = 1
        grads = {'x' : y}
        return grads

#Activation Function of Sinusoid:
class Sin(Activation_Function):
    def forward(self, x):
        y = sin(x)
        return y
    
    def backward(self, dY):
        return dY * self.grad['x']

    def local_grad(self, x):
        y = cos(x)
        grads = {'x' : y}
        return grads

#Activation Function of Tanh
class Tanh(Activation_Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, dY):
        return dY * self.grad['x']

    def local_grad(self, x):
        y = 1.0 - (np.tanh(x))**2
        grads = {'x' : y}
        return grads

#Activation Function of Sigmoid, Logistic, or softstep
class Sigmoid(Activation_Function):
    def forward(self, x):
        y = 1/(1+np.exp(-x))
        return y
    
    def backward(self, dY):
        return dY * self.grad['x']

    def local_grad(self, x):
        s=1/(1+np.exp(-x))
        y = s*(1-s)
        grads = {'x' : y}
        return grads

#Activation Function of Relu
class Relu(Activation_Function):
    def forward(self, x):
        if x<0.0:
            return 0
        else:
            return x
    
    def backward(self, dY):
        return dY * self.grad['x']

    def local_grad(self, x):
        if x<0.0:
            y =0
            grads = {'x' : y}
            return grads
        else:
            y = 1
            grads = {'x' : y}
            return grads

#Activation Function of Leaky Relu
######## alpha=0.01, 3 arguments ???
class Leaky_Relu(Activation_Function):
    def forward(self, x, alpha):
        if x<0.0:
                y = x * alpha
                return y
        else:
             return x

    def backward(self, dY):
        return dY * self.grad['x']

    def local_grad(self, x, alpha):
        if x<0.0:
            y = alpha
            grads = {'x' : y}
            return grads 
        else:
            y = 1
            grads = {'x' : y}
            return grads


#Activation Function of Hard Tanh
class Hard_Tanh(Activation_Function):
    def forward(self, x):
        if x <= -1.0:
            return -1
        elif 1.0<x<-1.0:
            return x
        else:
            return 1
        
    def backward(self, dY):
        return dY * self.grad['x']

    def local_grad(self, x):
        if x <= -1.0:
            y = 0
            grads = {'x' : y}
            return grads
        elif 1.0<x<-1.0:
            y = 1
            grads = {'x' : y}
            return grads
        else:
            y = 0
            grads = {'x' : y}
            return grads

#Activation Function of Softmax 
class Softmax(Activation_Function):
    def forward(self, x):
        exp_x = np.exp(x)
        y = exp_x / exp_x.sum(axis=0) 
        
        return y





fn=Relu()

print(fn.local_grad(-3))
print(fn.local_grad(0))
print(fn.local_grad(7))
print(fn.local_grad(-2))