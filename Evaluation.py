from modules.Layer import *
from modules.LOSS_FUNC import Cross_Entropy_Loss
from modules.Activation_Function import Relu,Softmax
from modules.Net import Net
from modules.PreProcessing_data import GetData
import numpy as np
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

Classes_number = 10
Predictions = []

for image,label in zip (Testing_data ,Testing_labels):

    image =image.reshape((1,1,28,28))
    out = LeNet(image)  #output of forward path

    preds = np.argmax(out, axis=1).reshape(-1, 1) #prediction of our Lenet mode
    Predictions.append(preds[0][0])



def evaluation(No_of_classes,True_Label,Predicted_Label):
    
    confussion_matrix = np.zeros((No_of_classes,No_of_classes))
    True_Label=(True_Label.T)[0] 
    for TL,PL in zip(True_Label,Predicted_Label):
        confussion_matrix[TL][PL]=confussion_matrix[TL][PL]+1

    TP = np.zeros(No_of_classes)
    TN = np.zeros(No_of_classes)
    FP = np.zeros(No_of_classes)
    FN = np.zeros(No_of_classes)
    for i in range(No_of_classes):
        TP[i] = confussion_matrix[i][i]
    Matrix_sum = np.sum(confussion_matrix)
    raw_sum = np.sum(confussion_matrix, axis = 1) 
    column_sum = np.sum(confussion_matrix,axis = 0)
    for i in range(No_of_classes):
        TN[i]= Matrix_sum-(raw_sum[i]+column_sum[i])+confussion_matrix[i][i]
    for i in range(No_of_classes):
        FP[i]= column_sum[i]-confussion_matrix[i][i]
    for i in range(No_of_classes):
        FN[i]=raw_sum[i]-confussion_matrix[i][i]

    accurecy =  np.sum(TP)/Matrix_sum
    accurecy=round(accurecy,2)
    Precision = np.zeros(No_of_classes)

    for i in range(No_of_classes):
        Precision[i]=TP[i]/(TP[i]+FN[i])
        Precision[i]=round(Precision[i],2)
    Recall = np.zeros(No_of_classes)

    for i in range(No_of_classes):
        Recall[i] = TP[i]/(TP[i]+TN[i])
        Recall[i]=round(Recall[i],2)
    print("TP:",TP)
    print("TN:",TN)
    print("FP:",FP)
    print("FN",FN)
    print("acc:",accurecy)
    print("Per:",Precision)
    print("Rec:",Recall)
    precision_Avg = round(np.sum(Precision)/No_of_classes,2)
    Recall_Avg = round(np.sum(Recall)/No_of_classes,2)

    print("Avg precision",precision_Avg)
    print("Avg ReCall",Recall_Avg)



evaluation(Classes_number,Testing_labels,Predictions)


