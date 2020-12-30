from Diff_Fun import Diff_Func
import numpy as np
from math import sqrt
from itertools import product
from utils import ZeroPadding


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
        self.Filter_Dim = Kernal_Size

        #Based on the (Filter_Dim, in_Channels, Num_Filters) we init our weights
        self.init_Weights()
        
        
        
    # Generate random values in a normal distribution for ( W ) - zeros for the Bias    
    def init_Weights(self):
        scale = 2/sqrt(self.in_Channels * self.Filter_Dim[0] * self.Filter_Dim[1])
        
        # W size (F , C , W , H)
        self.Weights= {'W' : np.random.normal(scale= scale , size= (self.Num_Filters , self.in_Channels ,self.Filter_Dim[0] ,self.Filter_Dim[1] ) )
                       'b' : np.zeros(shape= (self.Num_Filters , 1))    }
     
     
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
        
        Filter_W , Filter_H = self.Filter_Dim
        
        # Note : Padding Already Added from the Function Above so we dont need to add it here 
        W_out = ((W - Filter_W )/self.Stride) + 1
        H_out = ((H - Filter_H )/self.Stride) + 1
        
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
                    
                    Y[n , C_out , h , w ] = np.sum(self.Weights['W'][C_out] * Sub_X) + self.Weights['B'][C_out] 
                    
        
        return Y
                    
            
        
        
        
       
    
        
    
    
