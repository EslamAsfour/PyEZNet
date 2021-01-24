import numpy as np
from PyNNN.Diff_Func import Diff_Func


class Loss(Diff_Func):
    """
        Abstract Class to represent the Construction of the Loss Functions
    """
    def forward(self, X, Y):
        """
        Computes the loss of x with respect to y.
        Args:
        X: numpy.ndarray of shape (n_batch, n_dim).
        Y: numpy.ndarray of shape (n_batch, n_dim).
        Returns:
        loss: numpy.float.
        """
        pass
    def backward(self):
        """
        Backward pass for the loss function. Since it should be the final layer
        of an architecture, no input is needed for the backward pass.
        Returns:
        same grad value from calc_Grad()
        """
        return self.grad['X']
    def calc_Grad(self, X, Y):
        """
        Local gradient with respect to X at (X, Y).
        Args:
        X: numpy.ndarray of shape (n_batch, n_dim).
        Y: numpy.ndarray of shape (n_batch, n_dim).
        Returns:
        gradX: numpy.ndarray of shape (n_batch, n_dim).
        """
        pass


class Cross_Entropy_Loss(Loss):


    def forward(self, X, Y,total_loss="mean"):
        #X:is np array with shape (patch,dimen)
        #Y:is np array with shape (patch,1)
        #value of Y in every patch is the truth ground dimension.
        X_probs= Cross_Entropy_Loss.softmax(X)
        loss_array=np.zeros(Y.size)
        self.cache['probs']=X_probs
        for i in range (Y.size):
            #calculating the loss of every patch,prob. of the corresponding ground truth only as all other values will be *(0)
            loss_array[i]= -(np.log(X_probs[i,Y[i]]))
        if(total_loss=="mean"):
            return np.mean(loss_array)
        elif (total_loss=="sum"):
            return np.sum(loss_array)
        elif (total_loss =="median"):
            return np.median(loss_array)




    def calc_Grad(self, X, Y):
        #X:is np array with shape (patch,dimen)
        #Y:is np array with shape (patch,1)
        #value of Y in every patch is the truth ground dimension.
        '''
        probs=self.cache['probs']
        temp_grad= probs
        for i in range (Y.size):
            temp_grad[i,Y[i]]-=1
        grad={}
        grad['X']=temp_grad
        return grad
        '''
        probs = self.cache['probs']
        ones = np.zeros_like(probs)
        for row_idx, col_idx in enumerate(Y):
            ones[row_idx, col_idx] = 1.0

        grads = {'X': (probs - ones)/float(len(X))}
        return grads


    @staticmethod
    def softmax(X):
        X_exp = np.exp(X)
        sum_exp = np.sum(X_exp, axis=1, keepdims=True)
        return (X_exp / sum_exp)



class MeanSquareLoss(Loss):
    """
        Abstract Class to represent the Construction of the Loss Functions
    """
    def forward(self, X, Y):
        """
        X=output of layers
        Y=True labels
        """
        err = np.zeros(X.size)
        squ_sub = np.zeros(X.size)
        err = np.subtract(X,Y)
        squ_sub = np.square(err)
        sum = np.sum(squ_sub,axis=1 ,keepdims= True)
        total_loss = np.average(sum)

        return total_loss



    def calc_Grad(self, X, Y):
        """
        Local gradient with respect to X at (X, Y).
        Args:
        """
        grad = {}
        grad['X'] = 2*(X-Y)

        return grad

class Hinge_Loss(Loss):
    """
        Abstract Class to represent the Construction of the Hinge Loss Function used for binary classification.
    """
    def forward(self, X, Y,total_loss="average"):
        """
        Computes the loss of x with respect to y.
        Args:
        X: numpy.ndarray of shape (n_batch,1).
        Y: numpy.ndarray of shape (n_batch, 1).
        Returns:
        loss: numpy.float.
        """
        zeros_array=np.zeros(X.size)
        ones_array=np.ones(X.size)
        Loss_array=np.zeros(X.size)
        loss_mul=np.zeros(X.size)
        modified_mul=np.zeros(X.size)
        for i in range (X.size):
            loss_mul[i]=((Y[i])*(X[i]))
            modified_mul[i]=ones_array[i]-loss_mul[i]
        Loss_array=np.maximum(zeros_array,modified_mul)
        self.cache['mul']=loss_mul
        if(total_loss=="average"):
            return np.average(Loss_array)
        elif (total_loss=="sum"):
            return np.sum(Loss_array)
        elif (total_loss =="median"):
            return np.median(Loss_array)


    def calc_Grad(self, X, Y):
        """
        Computes the grad dictionary of x with respect to y.
        Args:
        X: numpy.ndarray of shape (n_batch,1).
        Y: numpy.ndarray of shape (n_batch, 1).
        Returns:
        grad: grad dictionary.
        """
        grad_mul=self.cache['mul']
        temp_grad=np.zeros(X.size)
        for i in range (X.size):
            if(grad_mul[i]<1):
                temp_grad[i]=-grad_mul[i]
            else:
                temp_grad[i]=0
        grad={}
        grad['X']=temp_grad
        return grad



