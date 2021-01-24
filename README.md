# PyEZNet
# Contents
 - [**Install our package**](#Install_our_package)
 - [**Potential Output**](#Example)
 - [**Modules**](#Modules)
     - [Layers](#Layers)
       - [`FullyConnected`](#FCD)
       - [`Conv2D`](#Conv2D)
       - [`MaxPooling`](#POOL)
       - [`AvgPooling`](#POOL)
       - [`Flatten`](#FLATTEN)
     - [Loss Functions](#Loss_functions)
       - [`CrossEntropy`](#CE_Loss)
       - `MeanSquareError`
       - `Hinge Loss`
     - [Activation Functions](#Activation_functions)
       - `ReLU`
       - `Leaky ReLU`
       - `Sigmoid`
       - `Softmax`
       - `Tanh`
       - `Hard Tanh`
       - `Linear`
   - [DataLoader](#DataLoader)
      - [`LoadingData`](#Loading_data)
      - [`Preprocessing Data`](#Preprocessing_data)
   - [Net](#Net)
   - [Evaluation](#Evaluation)


# Install our package<a name="Install_our_package"></a>
```python
pip install PyEZNet
```
-----
# Potential Output <a name="Example"></a>

This is an example for the output of a LeNet trained on a MNIST data set.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EslamAsfour/PyEZNet/blob/main/(Final)LeNet_Training.ipynb)


<p align="center">
  <img src="https://github.com/EslamAsfour/PyEZNet/blob/main/Diagrams-Docs/expected_output.gif" />
  </p>
  
-----


# Modules <a name="Modules"></a>

-----


# `Layers` <a name="Layers"></a>
## 1. Fully Connected :<a name="FCD"></a>

Fully Connected layer is used to take the output of convolution/pooling and predicts the best label to describe the image

<p align="center">
  <img src="https://github.com/EslamAsfour/PyEZNet/blob/main/Diagrams-Docs/fullyconnected.jpeg" />
  </p>

<br>

1.	We get the output from convolution/pooling and initialize our weights vector with the same dimension


```python

   def _init_Weights(self,input_dim,output_dim):
        scale = 1/ sqrt(input_dim)
                 self.weights['W'] = scale *np.random.randn(input_dim,output_dim)
                 self.weights['b'] = scale *np.random.randn(1,output_dim)
         
```





2.	Forward function take X 'vector of numpy array' where it's size is (no_of_batch,input_dim)and it's output is Y = (WX+b)


```python
  def forward(self,X):         output=np.dot(X,self.weights['W']) + self.weights['b']
            self.cache['X']=X
        self.cache['output']=output
        return output
```



3.	Our goal with backpropagation is to update each of the weights in the network, so that they cause the actual output to be closer the target output using 'Chain Rule'



```python

 def backward(self,dY):

            dX=dY.dot(self.grad['X'].T)
            X=self.cache['X']
            dw=self.grad['W'].T.dot(dY)
            db=np.sum(dY,axis=0,keepdims=True)          		  	self.weights_Update={'W': dw,'b': db}
        return dX
```



-----

## 2. Conv2D :<a name="Conv2D"></a>
### &nbsp;&nbsp;&nbsp;&nbsp;[1. Inputs , Outputs](#IO)
### &nbsp;&nbsp;&nbsp;&nbsp;[2. Forward Path Theoretically](#FPT)
### &nbsp;&nbsp;&nbsp;&nbsp;[3. Forward Path in Code](#FPIC)
### &nbsp;&nbsp;&nbsp;&nbsp;[4. Backward Path Theoretically](#BPT)
### &nbsp;&nbsp;&nbsp;&nbsp;[5. Backward Path in Code](#BPIC)

### 1. Inputs , Outputs<a name="IO"></a>
  
        
  ### Inputs for the layer :
   1.  in_Channels
   2.  out_Channels (Number of Filter in the Conv Layer)
   3.  Padding
   4.  Stride
   5.  Kernal Size (Size of the Filter ex: 3x3 filter)
  ### Conv2D Takes input img (Channel , Width , Height)  with N imgs -> (N , Ch , H , W) and Kernal Size 
  ### And we calculate the output size(H,W) by the formula :
  <p align="center">
  <img src="https://github.com/EslamAsfour/PyEZNet/blob/Conv2D-in-Dev/Diagrams-Docs/shape.png" />
  </p>



###  2. Forward Path Theoretically<a name="FPT"></a>
  ![alt text](https://github.com/EslamAsfour/PyEZNet/blob/Conv2D-in-Dev/Diagrams-Docs/Forward.gif)
  


###  3. Forward Path in Code<a name="FPIC"></a>
  ```python
    #loop over every Img
          for n in range(N):
              #Loop over every filter
              for C_out in range(self.Num_Filters):
                  #Loop over H & W
                  for h , w in product(range(W_out) , range(H_out)):
                      # Calc Starting index for every step based on the Stride
                      H_offset , W_offset = h*self.Stride , w*self.Stride
                      # Subset from X for the filter size
                      # Select [ one img (n)  , All the Ch , Offset+ Filter_h , Offset+Filter_W ]
                      Sub_X = X[n , : , H_offset: (H_offset + Filter_H) , W_offset: (W_offset + Filter_W)  ]
                      Y[n , C_out , h , w ] = np.sum(self.weights['W'][C_out] * Sub_X) + self.weights['b'][C_out]
  ```


###  4. Backward Path Theoretically<a name="BPT"></a>
   ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In the Backward we need to Calculate :
   ###   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. Grad X wrt L(Loss Func)
   ###   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. Grad W wrt L(Loss Func)
![alt text](https://github.com/EslamAsfour/PyEZNet/blob/Conv2D-in-Dev/Diagrams-Docs/Backward.gif)
 ### <div align="center">This GIF demonstrate the Calculation of Grad W </div>


  
  
###  5. Backward Path in Code<a name="BPIC"></a>


  ```python
    # Calc dX
        #Loop Over every img
        for n in range(N):
            # Loop over filters
            for C_out in range(self.Num_Filters):
                # Loop over h,w
                for h,w in product(range(dY.shape[2]),range(dY.shape[3])):
                    H_offset , W_offset = h*self.Stride , w*self.Stride
                    #                                                                                          filter index
                    dX[n,: , H_offset:H_offset + Filter_H , W_offset:W_offset + Filter_W ] += self.weights['W'][C_out] * dY[n,C_out,h,w]


        # Calc dW
        dW = np.zeros_like(self.weights['W'])

        # Loop over filters
        for c_w in range(self.Num_Filters):
            for c_i in range(self.in_Channels):
                for h,w in product(range(Filter_H),range(Filter_W)):
                    Sub_X = X[: , c_i , h:H-Filter_H+h+1:self.Stride  , w: W-Filter_W+w+1:self.Stride ]
                    dY_rec_field = dY[:, c_w]
                    dW[c_w, c_i, h ,w] = np.sum(Sub_X*dY_rec_field)
   ```

-----

## 3. Max Pooling & Average Pooling :<a name="POOL"></a>
### Why do we perform pooling? 
To reduce variance, reduce computation complexity (as 2*2 max pooling/average pooling reduces 75% data) and extract low level features from neighbourhood.

In this project we built:
- [x] Max pooling
- [x] Average Pooling 

Max pooling extracts the most important features like edges whereas, average pooling extracts features so smoothly. For image data, you can see the difference.
Although both are used for same reason, but max pooling is better for extracting the extreme features. Average pooling sometimes can’t extract good features because it takes all into count and results an average value which may/may not be important for object detection type tasks.

<p align="center">
  <img src="https://github.com/EslamAsfour/PyEZNet/blob/main/Diagrams-Docs/mpool.jfif" />
  </p>

-----

## `Loss Functions` <a name="Loss_functions"></a>
### Cross Entropy Loss: <a name="CE_Loss"></a>

Cross Entropy is used for multi-class classification, it takes three inputs:
1. X:The output of the fully connected layers, which corresponds to the prediction.
2. Y:The true output, which is zeros for all values except for the true label equals one.
3. The way of collecting total loss, either summing, or mean or average, the default is Mean.

**The Forward Function:**
starts by using softmax to generate probabilities for the input prediction array.
Then uses loss eqaution to calculate loss for every image which is: -sigma(log(X[i]*Y[i]) where i is used in for loop among the labels of each image.
After that it calculates total loss.
The function stores probabilities in the cache to be used in backward probagation.


**The Backward Function:**
Function returns The gradient values from the cache that was calculated by Calc_grad function.


**Calc_grad Function:**
it calculates grad using the euqation: q[i]−y[i] where q is the probabilities from softmax, y[i]=1 if i is the true label only and equal zero if otherwise.
then it stores gradient values in the cache to be used in backward probagation.


<p align="center">
  <img src="https://github.com/EslamAsfour/PyEZNet/blob/main/Diagrams-Docs/crossentropy.png" />
 </p>


-----
## `Activation Functions`<a name="Activation_functions"></a>

In this module we implement our Activation Functions and their derivatives:
We implement a class for each activation function. Each class contains three functions (forward function, backward functions and local_grad function). All of them inherit from Activation_Function class.<br>

Forward Function: we calculate the activation function itself.<br>

Backward Function: we use the gradient (derivative) in back propagation through layers.<br>

Local Gradient Function: we calculate the derivative of each function.<br>

Hard Tanh                        |Leaky ReLU                      |ReLU
:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/EslamAsfour/PyEZNet/blob/main/Diagrams-Docs/Hard_Tanh.jfif) |![](https://github.com/EslamAsfour/PyEZNet/blob/main/Diagrams-Docs/Leaky%20ReLU.jfif) |![](https://github.com/EslamAsfour/PyEZNet/blob/main/Diagrams-Docs/ReLU.jfif) |
Softmax                       |Tanh                      |Sigmoid
![](https://github.com/EslamAsfour/PyEZNet/blob/main/Diagrams-Docs/Softmax.jfif) |![](https://github.com/EslamAsfour/PyEZNet/blob/main/Diagrams-Docs/Tanh.jfif) |![](https://github.com/EslamAsfour/PyEZNet/blob/main/Diagrams-Docs/sigmoid.jfif) |

-----
## `DataLoader` <a name="DataLoader"></a>
### **1. Loading Data:**<a name="Loading_data"></a>

In this Dataloader script we download our dataset from "http://yann.lecun.com/exdb/mnist/".

Todo list:

1. we use "download_and_parse_mnist_file" function to download our files which is in .rar format in data directory and uncompress our files inside this function
2. it calls "parse_idx" function inside it to check that there is no error in it.
3. start print download progress
```python
def print_download_progress(count, block_size, total_size):
    pct_complete = int(count * block_size * 100 / total_size)
    pct_complete = min(pct_complete, 100)
    msg = "\r- Download progress: %d" % (pct_complete) + "%"
    sys.stdout.write(msg)
    sys.stdout.flush()
```
4. putting Training_Data , Tarining_labels , Testing_Data,Testing_lables in four seperated functions to get the from our files
```python
def train_images():
    return download_and_parse_mnist_file('train-images-idx3-ubyte.gz')


def test_images():
    return download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz')


def train_labels():
    return download_and_parse_mnist_file('train-labels-idx1-ubyte.gz')


def test_labels():
    return download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz')

```
### **2. Preprocessing Data:** <a name="Preprocessing_data"></a>
In this script we just :

1. Import our DataLoader script and use it to get our training and testing data with their labels.

2. Prepare our dataset by normalizing ,reshape and make them 2-D array 

```python
def GetData():
    print('Loadind data......')
    num_classes = 10
    train_images = DataLoader.train_images() #[60000, 28, 28]
    train_images= np.array(train_images,dtype=np.float32)
    train_labels = DataLoader.train_labels()
    test_images = DataLoader.test_images()
    test_images= np.array(test_images,dtype=np.float32)
    test_labels = DataLoader.test_labels()

    print('Preparing data......')
```
  3. Reshaping and normalization training dataset and its labels
```python
    training_data = train_images.reshape(60000, 1, 28, 28)
    training_data=training_data/255
    training_labels= train_labels.reshape (-1,1)

```
 4. Reshaping and normalization testing dataset and its labels
```python
    testing_data = test_images.reshape(10000, 1, 28, 28)
    testing_data=testing_data/255
    testing_labels=test_labels .reshape (-1,1)
    
    return training_data,training_labels,testing_data,testing_labels

```
-----

## `Net` <a name="Net"></a>
This script was used to test our CNN accuracy it consists of one class "Net"

1. We check that both of loss_fn and layer that will be the input to our CNN is one of our loss functions and layers 
"Conv_2D,MaxPooling,FC"

```python
def __init__(self,layers,loss):
        assert isinstance(loss,Loss) #the loss function must be as instance of nn.losses.Loss
        for layer in layers:
            assert isinstance(layer,Diff_Func)
            #layer must be instance of nn.layers.Layer or nn.layers.Function

        self.layers=layers
        self.loss_function=loss
```



2. Using Forward Path to go through the input layers one by one respectively.

```python
 def forward(self,x):
        '''
        :param x: numpy input array to our net
        :return:  numpy output array
        '''

        for layer in self.layers:
            x= layer(x)
        return x
```

3. Calculating our loss through the forward path 

```python
def loss(self,x,y):
        '''
        :param x: numpy array output of forward pass
        :param y: numpy array which is true values
        :return: numpy float loss value
        '''
        loss = self.loss_function(x,y)
        return loss
```

4. calculating backward path in order to decrease our loss and helping the model train


```python
 def backward(self):
        '''
        calculate backward pass for net which must be calculated after calculate forwad pass and loss
        :return: numpy array of shape matching the input during forward pass
        '''
        back=self.loss_function.backward()
        for layer in reversed(self.layers):
            back=layer.backward(back)
        return back

```
**(save_weights)** :

<br>

Saving the current weights in a pickle file ”.pkl” using the epoch and batch in the file name.
 We loop over the layers of the Net and save the weights calculated in each layer in that file.

<br>

**(load_weights)**:

<br>

Loading the weights that have been saved before in a “.pkl” file so we could use it again.
We loop over the layers of the Net and load the saved weights into the current weights of each layer in the net. 

<br>

-----

## `Evaluation` <a name="Evaluation"></a>

### The function's input:
- [ ] Number of classes (categories)
- [ ] True label
- [ ] Predicted label

### Then the function computes:
- [ ] The confusion matrix (TP, TN, FP, FN)
- [ ] Accuracy
- [ ] Precision
- [ ] Recall
- [ ] Average Precision
- [ ] Average Recall

### How we calculate the output?

1. TP (True Positive): The True Positives is the number of predictions where data labelled to belong to a particular class was correctly classified as the said class.

```python 
    for i in range(No_of_classes):
        TP[i] = confussion_matrix[i][i]
  ```
  
2. TN (True Negatives): The True Negative for a particular class is calculated by taking the sum of the values in every row and column except the row and column of the class we're trying to find the True Negatives for.

For example, calculating the True Negatives for the Greyhound class (assuming we have 3 classes: Greyhound, Mastiff, Samoyed):

<p align="center">
  <img src="https://github.com/EslamAsfour/PyEZNet/blob/main/Diagrams-Docs/tn1.jfif" />
  </p>


```python 
    for i in range(No_of_classes):
        TN[i]= Matrix_sum-(raw_sum[i]+column_sum[i])+confussion_matrix[i][i]
```

3. FP (False Positive): The False Positives for a particular class can be calculated by taking the sum of all the values in the column corresponding to that class except the True Positives value.

For example, calculating the False Positives for the Greyhound class (assuming we have 3 classes: Greyhound, Mastiff, Samoyed):

<p align="center">
  <img src="https://github.com/EslamAsfour/PyEZNet/blob/main/Diagrams-Docs/fp1.jfif" />
  </p>
  
```python
    for i in range(No_of_classes):
        FP[i]= column_sum[i]-confussion_matrix[i][i]
```

4. FN (False NEgative): The False Negatives for a particular class can be calculated by taking the sum of all the values in the row corresponding to that class except the True Positive values.

For example, calculating the False Negatives for the Greyhound class (assuming we have 3 classes: Greyhound, Mastiff, Samoyed):

<p align="center">
  <img src="https://github.com/EslamAsfour/PyEZNet/blob/main/Diagrams-Docs/fn1.jfif" />
  </p>

```python
    for i in range(No_of_classes):
        FN[i]=raw_sum[i]-confussion_matrix[i][i]
```

5. Accuracy =  TP / confusion matrix 

```python
    accurecy =  np.sum(TP)/Matrix_sum
```
 
6. Precision(class) = TP / (TP + FN)

```python
    for i in range(No_of_classes):
        Precision[i]=TP[i]/(TP[i]+FN[i])
```
 
7. Recall(class) = TP / (TP + TN)
 
 ```python
   for i in range(No_of_classes):
        Recall[i] = TP[i]/(TP[i]+TN[i])
```
 
 -----

