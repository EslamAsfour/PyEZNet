import numpy as np
from Diff_Func import Diff_Func
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
        return self.calc_Grad (X, Y)
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


    def forward(self, X, Y,total_loss="average"):
        #X:is np array with shape (patch,dimen)
        #Y:is np array with shape (patch,1)
        #value of Y in every patch is the truth ground dimension.
        X_probs= Cross_Entropy_Loss.softmax(X)
        loss_array=np.zeros(Y.size)
        self.cache['probs']=X_probs
        for i in range (Y.size):
            #calculating the loss of every patch,prob. of the corresponding ground truth only as all other values will be *(0)
            loss_array[i]= -(np.log(X_probs[i,Y[i]]))
        if(total_loss=="average"):
            return np.average(loss_array)
        elif (total_loss=="sum"):
            return np.sum(loss_array)
        elif (total_loss =="median"):
            return np.median(loss_array)




    def calc_Grad(self, X, Y):
        #X:is np array with shape (patch,dimen)
        #Y:is np array with shape (patch,1)
        #value of Y in every patch is the truth ground dimension.
        probs=self.cache['probs']
        temp_grad=probs
        for i in range (Y.size):
            temp_grad[i,Y[i]]-=1
        grad=temp_grad
        return grad


    @staticmethod
    def softmax(X):
        X_exp = np.exp(X)
        sum_exp = np.sum(X_exp, axis=1, keepdims=True)
        return (X_exp / sum_exp)





########################## Test case################################

list1=[[1,2,3,4,5,6],[6,2,3,4,6,7],[3,5,7,8,3,5],[1,5,8,4,5,6],[1,2,6,4,8,6]]
list2=[5,3,5,2,1]
X=np.array(list1)
Y=np.array(list2)
loss1=Cross_Entropy_Loss()
foro=loss1.forward(X,Y)
print(foro)
back=loss1.backward()
print(back)
