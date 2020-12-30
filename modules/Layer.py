from Diff_Fun import Diff_Func


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
        
        
        
        
    def init_Weights()
    
    
    
