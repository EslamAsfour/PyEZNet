from modules.Layer import *
from modules.LOSS_FUNC import Cross_Entropy_Loss
from modules.Activation_Function import Relu,Softmax
from modules.Net import Net
from modules.PreProcessing_data import GetData

#Todo

'''
1-create object from LeNet
2-Loading our Pretrained Weights
3-Load our Testing dataset
4-visualize our loss
'''

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


path = "logs\Weights_(1)_(700).pkl"
LeNet.load_weights(path)

Training_data,Training_labels,Testing_data , Testing_labels = GetData()

Classes_name = [0,1,2,3,4,5,6,7,8,9]
Predictions = []

for image,label in zip (Testing_data ,Testing_labels):

    image =image.reshape((1,1,28,28))
    out = LeNet(image)  #output of forward path

    preds = np.argmax(out, axis=1).reshape(-1, 1) #prediction of our Lenet mode
    Predictions.append(preds[0][0])

print(Predictions)
print((Testing_labels))






