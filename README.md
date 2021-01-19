# Custom_DL_Framework-Project

## What is the Target Output ?

```python
  net = Net(layers=[Linear(2, 4), ReLU(), Linear(4, 2)],
          loss=CrossEntropyLoss())

  out = net(X)
  loss = net.loss(out, Y_labels)
  grad = net.backward()
  net.update_weights(lr=0.1)
```



<br>
<br>
<br>
<br>

## Our Modules 

![alt text](https://github.com/EslamAsfour/Custom_DL_Framework-Project/blob/main/Diagrams-Docs/Custom_DL_Framework%20Project%20Diagram.png)
