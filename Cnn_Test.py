import numpy as np

from modules.Layer import *
from modules.LOSS_FUNC import Cross_Entropy_Loss
from modules.Activation_Function import Relu
from modules.Net import Net
from modules.Untracked import BatchNorm2D


from keras.datasets import mnist

np.random.seed(1)


net = Net(layers=[Conv2D(1, 4, Kernal_Size= 3, Padding=1), MaxPool2D(kernel_size=2), Relu(), BatchNorm2D(4),
                  Conv2D(4, 8, Kernal_Size= 3, Padding=1), MaxPool2D(kernel_size=2), Relu(), BatchNorm2D(8),
                  Flatten(), FullyConnectedLayer(8*7*7, 10)],
          loss=Cross_Entropy_Loss())

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshaping
X_train, X_test = X_train.reshape(-1, 1, 28, 28), X_test.reshape(-1, 1, 28, 28)
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
# normalizing and scaling data
X_train, X_test = X_train.astype('float32')/255, X_test.astype('float32')/255

n_epochs = 1000
n_batch = 100
for epoch_idx in range(n_epochs):
    batch_idx = np.random.choice(range(len(X_train)), size=n_batch, replace=False)
    out = net(X_train[batch_idx])
    preds = np.argmax(out, axis=1).reshape(-1, 1)
    accuracy = 100*(preds == y_train[batch_idx]).sum() / n_batch
    loss = net.loss(out, y_train[batch_idx])
    net.backward()
    net.weights_update(alpha=0.01)
    print("Epoch no. %d loss =  %2f4 \t accuracy = %d %%" % (epoch_idx + 1, loss, accuracy))
