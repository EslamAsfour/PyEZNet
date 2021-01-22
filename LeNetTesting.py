from modules.Layer import *
from modules.LOSS_FUNC import Cross_Entropy_Loss
from modules.Activation_Function import Relu,Softmax
from modules.Net import Net
from modules.PreProcessing_data import GetData
import cv2
import matplotlib.pyplot as plt

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

while (1):
    No_ofImage =input("Image Number ")
    if (not No_ofImage.isdigit()):
        break
    No_ofImage = int(No_ofImage)
    Testing_image=Testing_data[No_ofImage]
    Testing_image =Testing_image.reshape((1,1,28,28))
    Image_labels=Testing_labels[No_ofImage]

    #Forwarding path
    out = LeNet(Testing_image)

    #prediction of our Lenet model
    preds = np.argmax(out, axis=1).reshape(-1, 1)

    #output our model prediction
    print("LeNet Prediction is {}".format(preds[0][0]))

    #showing image
    Testing_image=Testing_image.reshape(28,28)
    plt.imshow(Testing_image,cmap='gray')
    plt.show()






