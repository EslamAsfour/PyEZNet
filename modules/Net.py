from modules.LOSS_FUNC import Loss
from modules.Diff_Func import Diff_Func
from modules.Layer import Layer
import numpy as np
import pickle


class Net:
    #memory optimization for storage
    __slots__ = ['layers','loss_function']

    # layers and loss are parameter
    #while argument is the value sent to the function when it is called
    def __init__(self,layers,loss):
        assert isinstance(loss,Loss) #the loss function must be as instance of nn.losses.Loss
        for layer in layers:
            assert isinstance(layer,Diff_Func)
            #layer must be instance of nn.layers.Layer or nn.layers.Function

        self.layers=layers
        self.loss_function=loss

    def __call__(self, *args, **kwargs):
        '''
        :param args: don't know the no.of arg fe 3mlt * ablha
        :param kwargs: keyword argument w bardo msh 3arf 3ddhum fe 3mltha **
        :return: self.forward
        '''
        return self.forward(*args, **kwargs)

    def forward(self,x):
        '''
        :param x: numpy input array to our net
        :return:  numpy output array
        '''

        for layer in self.layers:
            x= layer(x)
        return x

    def loss(self,x,y):
        '''
        :param x: numpy array output of forward pass
        :param y: numpy array which is true values
        :return: numpy float loss value
        '''
        loss = self.loss_function(x,y)
        return loss


    def backward(self):
        '''
        calculate backward pass for net which must be calculated after calculate forwad pass and loss
        :return: numpy array of shape matching the input during forward pass
        '''
        back=self.loss_function.backward()
        for layer in reversed(self.layers):
            back=layer.backward(back)
        return back


    def weights_update(self,alpha):
        '''
        updates the weights using the gradient which is calculated during backpropagation
        :param alpha: numpy float number
        :return:
        '''
        for layer in self.layers:
            if isinstance(layer,Layer):
                layer.Update_Weights(alpha)
    def save_weights(self,epoch,patch):
        i=0
        file_name = "Logs/Weights_("+ str(epoch) + ")_(" + str(patch) + ").pkl"
        obj = []
        for l in self.layers:
            if isinstance(l,Layer):
                cache = l.weights
                obj.append(cache)
            i+=1
        with open(file_name, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

            i+=1
    def load_weights(self,path):
        
        with open(path, 'rb') as f:
            dic = pickle.load(f)

        i=0
        for l in self.layers:
            if isinstance(l,Layer) :
                l.weights['W'] = dic[i]['W'] 
                l.weights['b'] = dic[i]['b']
                i+=1
                