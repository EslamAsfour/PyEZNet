import os
import numpy as np
from PIL import Image



def zero_pad(X,pad_width,dims):
    '''
    pad the given array x with zeros at both end of given dim
    :param X: numpy array
    :param pad_width: width of padding
    :param dims: dimensions to be padded
    :return: numpy array with zero padding X

    '''
    dims = (dims) if isinstance(dims,int) else dims
    '''
    isinstance return true if dims is int so dims = tuple dims else will be equal itself
    '''
    pad=[(0,0) if idx not in dims else (pad_width,pad_width)
         for idx in range(len(X.shape))]
    x_padded=np.pad(X,pad,'constant')
    return x_padded






def data_loader(folderpath):
    images= []
    lables= []
    for class_dir in os.listdir(folderpath):
        class_labels = int(class_dir)-1
        class_path = os.path.join(folderpath,class_path)
        for fname in os.listdir(class_path):
            images.append(np.array(Image.open(os.path.join(class_path,fname))).transpose(2,0,1))

        lables.append(np.array([class_labels]*len(os.listdir(class_path))))
    X=np.concatenate(images,axis=0)
    y=np.concatenate(lables).reshape(-1,1)
    return X,y





