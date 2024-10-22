import numpy
import numpy as np
import math
from math import exp
from PyEZNet.Diff_Func import Diff_Func


'''Activation Functions:
       1. Linear or identity
       2. Sinusoid
       3. Tanh
       4. Sigmoid, Logistic, or softstep
       5. Relu
       6. Leaky Relu
       7. Hard Tanh
'''

#Activation Function of linear or identity
class Linear(Diff_Func):
    def forward(self, x):
        return x

    def backward(self, dY):
        return dY * self.grad['x']


    def calc_Grad(self, x):
        y = 1
        grads = {'x' : y}
        return grads

#Activation Function of Sinusoid:
class Sin(Diff_Func):
    def forward(self, x):
        y = sin(x)
        return y
    
    def backward(self, dY):
        return dY * self.grad['x']

    def calc_Grad(self, x):
        y = cos(x)
        grads = {'x' : y}
        return grads

#Activation Function of Tanh
class Tanh(Diff_Func):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, dY):
        return dY * self.grad['x']

    def calc_Grad(self, x):
        y = 1.0 - (np.tanh(x))**2
        grads = {'x' : y}
        return grads

#Activation Function of Sigmoid, Logistic, or softstep
class Sigmoid(Diff_Func):
    def forward(self, x):
        y = 1/(1+np.exp(-x))
        return y
    
    def backward(self, dY):
        return dY * self.grad['x']

    def calc_Grad(self, x):
        s=1/(1+np.exp(-x))
        y = s*(1-s)
        grads = {'x' : y}
        return grads

#Activation Function of Relu
class Relu(Diff_Func):
    def forward(self, x):
        return x*(x > 0)
    
    def backward(self, dY):
        return dY * self.grad['X']

    def calc_Grad(self, x):
        
        grads = {'X': 1*(x > 0)}
        return grads

#Activation Function of Leaky Relu
######## alpha=0.01, 3 arguments ???
class Leaky_Relu(Diff_Func):
    def forward(self, x, alpha):
        if x<0.0:
                y = x * alpha
                return y
        else:
             return x

    def backward(self, dY):
        return dY * self.grad['x']

    def calc_Grad(self, x, alpha):
        if x<0.0:
            y = alpha
            grads = {'x' : y}
            return grads 
        else:
            y = 1
            grads = {'x' : y}
            return grads


#Activation Function of Hard Tanh
class Hard_Tanh(Diff_Func):
    def forward(self, x):
        if x <= -1.0:
            return -1
        elif 1.0<x<-1.0:
            return x
        else:
            return 1
        
    def backward(self, dY):
        return dY * self.grad['x']

    def calc_Grad(self, x):
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
class Softmax(Diff_Func):
    
    def forward(self, x):
        out = np.zeros(shape=(32,10))
        x -= np.max(x)
        for i in range(x.shape[0]):
            out[i] = (np.exp(x[i]).T / np.sum(np.exp(x[i]), axis=0)).T
        self.y = out
        return self.y
    

    def backward(self, dY):
        grad = np.zeros(shape=(1,10))
        out = np.zeros(shape=(32,10))
        for i in range(dY.shape[0]):
            s = self.y[i].reshape(-1, 1)
            grad = np.diagflat(s) - np.dot(s, s.T)
            out[i] = np.dot(grad  , dY[i])
            
        return out

    def local_grad(self, x):
        pass
    '''
    def forward(self, x):
        exp_x = np.exp(x)
        y = exp_x / exp_x.sum(axis=0) 
        
        return y

    def backward(self, dY):
        return dY * self.grad['x']


    def local_grad(self, x):
        exp_x = np.exp(x)
        y = exp_x / exp_x.sum(axis=0) 
        # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
        s = y.reshape(-1,1)
        gradient=np.diagflat(s) - np.dot(s, s.T)
        grads={'x': gradient}
        return grads
    '''




