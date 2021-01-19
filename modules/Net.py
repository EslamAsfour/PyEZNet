# from .losses import Loss
#from .layers import Function , Layer

class Net:
    #memory optimization for storage
    __slots__ = ['layers','loss_function']

    # layers and loss are parameter
    #while argument is the value sent to the function when it is called
    def __init__(self,layers,loss):
        assert isinstance(loss,Loss) #the loss function must be as instance of nn.losses.Loss
        for layer in layers:
            assert isinstance(layer,Function)
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
            back=layer.backward()
        return back


    def weights_update(self,alpha):
        '''
        updates the weights using the gradient which is calculated during backpropagation
        :param alpha: numpy float number
        :return:
        '''
        for layer in self.layers:
            if isinstance(layer,Layer):
                layer._update_weights(alpha)