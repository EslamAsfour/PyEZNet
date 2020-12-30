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
    """
        Abstract Class to represent the Construction of the Multi class cross entropy Loss Function used for multi class classification.
    """

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
        grad={}
        grad['X']=temp_grad
        return grad


    @staticmethod
    def softmax(X):
        X_exp = np.exp(X)
        sum_exp = np.sum(X_exp, axis=1, keepdims=True)
        return (X_exp / sum_exp)


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






########################## Test case for Cross Entropy################################
print("############################# Cross Entropy loss test ############################\n")
list1=[[1,2,3,4,5,6],[6,2,3,4,6,7],[3,5,7,8,3,5],[1,5,8,4,5,6],[1,2,6,4,8,6]]
list2=[5,3,5,2,1]
X=np.array(list1)
Y=np.array(list2)
print("X:",X,"\n")
print("Y:",Y,"\n")

loss1=Cross_Entropy_Loss()
foro=loss1(X,Y)
print("Total cross entropy loss:",foro,"\n")
back=loss1.backward()
print("Gradient array of cross entropy loss:\n",back,"\n")



##################################### Test case for Hinge loss ########################
print("############################# Hinge loss test ############################\n\n")
list2=[-5,3,-5,2,-1]
list1=[2,5,7,9,5]
X=np.array(list1)
Y=np.array(list2)
print("X:",X,"\n")
print("Y:",Y,"\n")
loss2=Hinge_Loss()
forward=loss2(X,Y)
print("Total Hinge loss:",forward,"\n")
backward=loss2.backward()
print("Gradient array of Hinge loss:\n",backward,"\n")
