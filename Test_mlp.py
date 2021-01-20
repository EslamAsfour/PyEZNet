import numpy as np

from modules.Layer import *
from modules.LOSS_FUNC import Cross_Entropy_Loss
from modules.Activation_Function import Relu,Softmax
from modules.Net import Net
# functions for visualization

np.random.seed(0)


# generating some data
n_class_size = 100
r = 2
X1_offset = np.random.rand(n_class_size, 2) - 0.5
np.sqrt(np.sum(X1_offset**2, axis=1, keepdims=True))
X1_offset = r * X1_offset/np.sqrt(np.sum(X1_offset**2, axis=1, keepdims=True))
X1 = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], size=n_class_size) + X1_offset
X2 = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], size=n_class_size)

X = np.concatenate((X1, X2))
Y_labels = np.array([0]*n_class_size + [1]*n_class_size)

net = Net(layers=[FullyConnectedLayer(2, 4), Relu(), FullyConnectedLayer(4, 2)],
          loss=Cross_Entropy_Loss())

print(net)
n_epochs = 1000
for epoch_idx in range(n_epochs):
    print("Epoch no. %d" % epoch_idx)
    out = net(X)
    print(out)
    # prediction accuracy
    pred = np.argmax(out, axis=1)
    print("accuracy: %1.4f" % (1 - np.abs(pred - Y_labels).sum()/200))
    loss = net.loss(out, Y_labels)
    print('loss: %1.4f' % loss)
    grad = net.backward()
    net.weights_update(0.1)
    
       