import numpy as np
import cv2 

''' 
    Zero Padding Function 
'''

def ZeroPadding(X ,pad_Width ,dims):
    
    #if dims == (2,3)
    #   pad =[(0,0)(0,0)(P_D,P_D)(P_D,P_D)]
    pad = [(0, 0) if idx not in dims else (pad_width, pad_width) for idx in range(len(X.shape))]
    X_padded = np.pad(X, pad, 'constant')
    
    return X_padded
    
    
'''    
pad_width = 2    
dims = (2,3) 
X =   cv2.imread("E:/AUV/DataSet Creation/Final DS/test/2.jpg")
print(X.shape)  
pad = [(0, 0) if idx not in dims else (pad_width, pad_width) for idx in range(len(X.shape))]
X_padded = np.pad(X, pad, 'constant')

print(pad)
print(X_padded)
cv2.imshow("hH",X_padded )
cv2.waitKey(0)
print(pad)
'''