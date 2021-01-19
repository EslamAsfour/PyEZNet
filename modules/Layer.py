from modules.Diff_Func import Diff_Func
import numpy as np
from math import sqrt
from itertools import product
from modules.utils import ZeroPadding


class Layer (Diff_Func):
    '''
    Abstract Class to represent Layers so we added 
        1- Weights dic
        2- Weights update dic
        3- Init weights Function
        4- Update_Weights Function    
    '''    
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
         
        for weigh_key , weight in self.Weights.item():
            self.Weights[weight_key] = self.Weights[weight_key] - learningRate * self.Weights_Update[weight_key]

class MaxPool2D(Function):
###############################################################################
#
# Class: MaxPool2D
#
# File Name: layer.py
#
# Description: class for the maximum pooling layer.
# 
# Author: Ahmed Gamal
################################################################################
    def __init__(self, kernel_size=(2, 2)):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        #isinstance ->  check if the kernel size is int number 
    def __call__(self, X):
        # in contrary to other Function subclasses, MaxPool2D does not need to call
        # .local_grad() after forward pass because the gradient is calculated during it
        return self.forward(X)
    def forward(self, X):
        N, C, H, W = X.shape
        KH, KW = self.kernel_size

        grad = np.zeros_like(X)
        Y = np.zeros((N, C, H//KH, W//KW)) #h/kh -> hight of the new matrix ,, w/kw -> width of the new matrix

        for h, w in product(range(0, H//KH), range(0, W//KW)):
            h_offset, w_offset = h*KH, w*KW
            rec_field = X[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW] # get the part of matrix that we will find it max 
            Y[:, :, h, w] = np.max(rec_field, axis=(2, 3)) #find the max in the part of matrix
            for kh, kw in product(range(KH), range(KW)):
                grad[:, :, h_offset+kh, w_offset+kw] = (X[:, :, h_offset+kh, w_offset+kw] >= Y[:, :, h, w])

        # storing the gradient
        self.grad['X'] = grad

        return Y
    def backward(self, dY):
        dY = np.repeat(np.repeat(dY, repeats=self.kernel_size[0], axis=2),
                       repeats=self.kernel_size[1], axis=3)
        return self.grad['X']*dY

    def local_grad(self, X):
        # small hack: because for MaxPool calculating the gradient is simpler during
        # the forward pass, it is calculated there and this function just returns the
        # grad dictionary
        return self.grad   

class Conv2D(Layer):
    """
        Conv2D Takes input img (Channel , Width , Height)  with N imgs -> (N , Ch , H , W)
        and Kernal Size 
        
        Inputs for the layer :
            1-  in_Channels
            2-  out_Channels (Number of Filter in the Conv Layer)
            3-  Padding
            4-  Stride
            5-  Kernal Size (Size of the Filter ex: 3x3 filter)
    """
    def __init__(self , in_Channels , out_Channels , Padding=0 , Stride=1 , Kernal_Size=3):
        # Super Class will creat 
        #    ( grad , cache , Weights , Weights_Update ) Empty local dicts
        super().__init__()

        self.in_Channels = in_Channels
        self.Num_Filters = out_Channels
        self.Pad = Padding
        self.Stride = Stride
        self.Filter_Dim = (Kernal_Size,Kernal_Size)

        #Based on the (Filter_Dim, in_Channels, Num_Filters) we init our weights
        self.init_Weights()
             
    # Generate random values in a normal distribution for ( W ) - zeros for the Bias    
    def init_Weights(self):
        scale = 2/sqrt(self.in_Channels * self.Filter_Dim[0] * self.Filter_Dim[1])
        
        # W size (F , C , W , H)
        #self.Weights= {'W' : np.random.normal(scale= scale , size= (self.Num_Filters , self.in_Channels ,self.Filter_Dim[0] ,self.Filter_Dim[1] ))
        #               ,'b' : np.zeros(shape= (self.Num_Filters , 1)) }
        self.Weights= {'W' : [[[[-0.2392, -0.0492, -0.0347],
                             [-0.3049, -0.1630,  0.1242],
                             [-0.2988, -0.3229,  0.1064]]]] ,
                       'b' : [-0.0967] }
    
    
    def forward(self, X):
        '''
            Input : 
                - X -> Batch of imgs or single img --   X.shape = (N , Ch , H , W)
            Output :
                - Y -> Output of the Forward Prop with the current W  --  Y.shape = (N , F , H_out , W_out)
                Note : 
                    W_out and H_out = floor(( W_in + 2* padding - kernal_size )/ stride + 1)        
        '''
        # if we have Padding we first adjust the X 
        if self.Pad != 0:
            #Dims always (2,3) H,W 
            X = ZeroPadding(X , self.Pad , dims=(2,3))
            
        #Save input for backProp
        self.cache ={'X' : X}
        
        N , Ch , H , W = X.shape
        
        Filter_H , Filter_W = self.Filter_Dim
        
        # Note : Padding Already Added from the Function Above so we dont need to add it here 
        W_out = int(((W - Filter_W )/self.Stride) + 1)
        H_out = int(((H - Filter_H )/self.Stride) + 1)
        
        Y = np.zeros(shape=(N , self.Num_Filters , H_out ,W_out ))
        
        #loop over every Img
        for n in range(N):
            #Loop over every filter
            for C_out in range(self.Num_Filters):
                #Loop over H & W
                for h , w in product(range(W_out) , range(H_out)):
                    H_offset , W_offset = h*self.Stride , w*self.Stride
                    # Subset from X for the filter size
                    # Select [ one img (n)  , All the Ch , Offset+ Filter_h , Offset+Filter_W ]
                    Sub_X = X[n , : , H_offset: (H_offset + Filter_H) , W_offset: (W_offset + Filter_W)  ]
                    
                    Y[n , C_out , h , w ] = np.sum(self.Weights['W'][C_out] * Sub_X) + self.Weights['b'][C_out]     
        return Y
    def backward(self , dY):
        '''
            The Goal is to calc :
                1- dW/dL    -> Conv(X , dY/dl)
                2- dB/dL        Conv(X , dY/dl)
                3- dX/dL
        ''' 
        # Load X from the cache
        X = self.cache['X']
        # dX same shape as X
        dX = np.zeros_like(X)
        N ,CH ,H ,W = X.shape
        
        Filter_H , Filter_W = self.Filter_Dim
        
        # Calc dX
        #Loop Over every img
        for n in range(N):
            # Loop over filters
            for C_out in range(self.Num_Filters):
                # Loop over h,w
                for h,w in product(range(dY.shape[2]),range(dY.shape[3])):
                    H_offset , W_offset = h*self.Stride , w*self.Stride
                    #                                                                                          filter index
                    dX[n,C_out, H_offset:H_offset + Filter_H , W_offset:W_offset + Filter_W ] += self.Weights['W'][C_out] * dY[n,C_out,h,w]
                    
        # Calc dW
        dW = np.zeros_like(self.Weights['W'])
        
        # Loop over filters
        for c_w in range(self.Num_Filters):
            for c_i in range(self.in_Channels):
                for h,w in product(range(Filter_H),range(Filter_W)):
                    ######################################################## Msh fahm #####################################################
                    Sub_X = X[: , c_i , h:H-Filter_H+h+1:self.stride  , w: W-Filter_W+w+1:self.stride ]
                    dY_rec_field = dY[:, c_w]
                    dW[c_w, c_i, h ,w] = np.sum(Sub_X*dY_rec_field)
                    ##################################################### #####################################################
                        
        # Calc dB
        ######################### Msh Fahmm #############################
        dB = np.sum(dY, axis=(0, 2, 3)).reshape(-1, 1)
        ######################################################
        # caching the global gradients of the parameters
        self.weight_update['W'] = dW
        self.weight_update['b'] = db
        
        
        return dX[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        
        
        
       
    
        
    
    
