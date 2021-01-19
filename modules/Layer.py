from Diff_Fun import Diff_Func
import numpy as np
from math import sqrt



class Layer (Diff_Func):
"""
    Abstract Class to represent Layers so we added 
        1- Weights dic
        2- Weights update dic
        3- Init weights Function
        4- Update_Weights Function    
"""    
    def __init__(self, *args, **kwargs):
        # Call Diff_Func init function
        super().__init__(*args, **kwargs)
        # Empty Weights and updates
        self.Weights = {}
        self.Weights_Update = {}
        
     # Function to initalize weights   
     def init_Weights(self , *args, **kwargs):
         pass
     
     def Update_Weights(self , learningRate):
         
         for weigh_key , weight in self.Weights.item()
            self.Weights[weight_key] = self.Weights[weight_key] - learningRate * self.Weights_Update[weight_key]



class Conv2D(Layer):
    """
        Conv2D Takes input img (Channel , Width , Height)  with N imgs -> (N , Ch , W , H)
        and Kernal Size 
        
    
    """
    

class FullyConnectedLayer(Layer):

     def __init__self(self,input_dim,output_dim):
         self().__init__()      #inheret initial function from our parent class 'Layer'
         self._init_weights(input_dim,output_dim)
         self.cache = {}
         self.weight_update = {}

     def _init_weights(self,input_dim,output_dim):
         scale = 1/ sqrt(input_dim)
         #input_dim= rows , output_dim = columns
         self.weights['W'] = scale *np.random.randn(input_dim,output_dim)
         # 1 = rows , output_dim = columns
         self.weights['b'] = scale *np.random.randn(1,output_dim)


     def Forward(self,X):

    # forwar function take X 'vector of numpy array' where it's size is (no_of_batch,input_dim)
    # and it's output is Y = (WX+b)
        output=np.dot(X,self.Weights['W']) + self.Weights['b']
    #cashe it's like a dictionary to save the answer of a frequent qeustions
        self.cache['X']=X
        self.cache['output']=output
        return output

     def Backword(self,dY):
    # Our goal with backpropagation is to update each of the weights in the network
    # so that they cause the actual output to be closer the target output using 'Chain Rule'
        dX=dY.dot(self.grad['X'].T)
    # dY is output array of size (rows= no-of_batches, colomns = no_of_out)
    # calculate golbal gradient to be used in backpropagation

        X=self.cache['X']
        dw=self.grad(['W']).T.dot(dY)
    # calculating the global gradient wrt to weight
        db=np.sum(dY,axis=0,keepdims=True)
    # keepdims=True means that the result will broadcast correctly against the input array.
    # axis=0 means will sum all of the elements of the input array.
    # dY is the elements to sum
        self.weight_update={'W':dw,['b']:db}
        return dX


     def grad(self,X):

         gradX_local = self.weight['W'] #weight hena gaya mn class function el kber el lsa hyt3mlo implementation
         gradW_local= X
         gradb_local = np.ones_like(self.weight['b'])
         grads = {'X': gradX_local, 'W': gradW_local, 'b': gradb_local}
         return grads


