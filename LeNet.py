import numpy as np

from modules.Layer import *
from modules.LOSS_FUNC import Cross_Entropy_Loss
from modules.Activation_Function import Relu,Softmax
from modules.Net import Net
from modules.PreProcessing_data import GetData


# Load Data

X_train,y_train,X_test, y_test = GetData()

LeNet = Net(layers=[
                    Conv2D(in_Channels= 1,out_Channels = 6, Kernal_Size= 5, Padding=2  ,Stride= 1),
                    Relu(), 
                    MaxPool2D(kernel_size=2), 
                    Conv2D(in_Channels= 6,out_Channels = 16, Kernal_Size= 5, Padding=0  ,Stride= 1),
                    Relu(),
                    MaxPool2D(kernel_size=2),
                    Conv2D(in_Channels= 16,out_Channels = 120, Kernal_Size= 5, Padding=0  ,Stride= 1),
                    Relu(),
                    Flatten(),
                    FullyConnectedLayer(input_dim= 120,output_dim=84),
                    Relu(),
                    FullyConnectedLayer(input_dim=84,output_dim=10),
                    ], loss=Cross_Entropy_Loss())

LeNet.save_weights()

LeNet.load_weights()

n_epoch = 10
batch_size = 32


for e in range(n_epoch):
    
    for batch_index in range(0, X_train.shape[0], batch_size):
        if batch_index + batch_size < X_train.shape[0]:
            end_Index =   batch_index+batch_size
            x = X_train[ batch_index : end_Index ]
            y = y_train[ batch_index : end_Index ]
            
        else :
            end_Index =   X_train.shape[0]
            x = X_train[batch_index : end_Index]
            y = y_train[batch_index: end_Index]
        
        # Forward Prop
        out = LeNet(x)
        # Calc Accuracy
        '''
        preds = np.argmax(out, axis=1).reshape(-1, 1)
        accuracy = 100*( preds == y ).sum() / batch_size
        '''
        # Calc Loss
        loss = LeNet.loss(out, y)
        
        
        # Backward Prop
        LeNet.backward()
        LeNet.weights_update(alpha=0.01)
        
        print("Epoch no. %d loss =  %2f4 \t accuracy = %d %%" % ( e + 1, loss, 1))
        
        
        
        
        
    
    




