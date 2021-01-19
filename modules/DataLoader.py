import os
import numpy as np
from PIL import Image


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





