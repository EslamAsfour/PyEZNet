# Conv2D Layer
### [1. Inputs , Outputs](#IO)
### [2. Forward Path Theoretically](#FPT)
### [3. Forward Path in Code](#FPIC)
### [4. Backward Path Theoretically](#BPT)
### [5. Backward Path in Code](#BPIC)


## 1. Inputs , Outputs<a name="IO"></a>
  
        
  #### Inputs for the layer :
   1.  in_Channels
   2.  out_Channels (Number of Filter in the Conv Layer)
   3.  Padding
   4.  Stride
   5.  Kernal Size (Size of the Filter ex: 3x3 filter)
  #### Conv2D Takes input img (Channel , Width , Height)  with N imgs -> (N , Ch , H , W) and Kernal Size 
  #### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;And we calculate the output size(H,W) by the formula :
  <p align="center">
  <img src="https://github.com/EslamAsfour/Custom_DL_Framework-Project/blob/Conv2D-in-Dev/Diagrams-Docs/shape.png" />
  </p>



##  2. Forward Path Theoretically<a name="FPT"></a>
  ![alt text](https://github.com/EslamAsfour/Custom_DL_Framework-Project/blob/Conv2D-in-Dev/Diagrams-Docs/Forward.gif)
  


##  3. Forward Path in Code<a name="FPIC"></a>
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


##  4. Backward Path Theoretically<a name="BPT"></a>
   ### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In the Backward we need to Calculate :
   ####   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. Grad X wrt L(Loss Func)
   ####   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. Grad W wrt L(Loss Func)
![alt text](https://github.com/EslamAsfour/Custom_DL_Framework-Project/blob/Conv2D-in-Dev/Diagrams-Docs/Backward.gif)
 #### <div align="center">This GIF demonstrate the Calculation of Grad W </div>


  
  
##  5. Backward Path in Code<a name="BPIC"></a>


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

<br>
