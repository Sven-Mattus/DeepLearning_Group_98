import numpy as np

def tanh(x):
    tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return tanh

def softmax(x):
    #implementation of softmax, for now due to numerical stability taken from
    #desertnaut, aksed 23 Jan 2016, edited 20 Oct 2023, visited on 02.05.2024
    #https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    
    softmax = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return softmax

#alternative for softmax for numerical stability
#def softmax(self,x):
#    p = np.exp(x - np.max(x))
#    return p / np.sum(p)

#a[t] = np.dot(W, h[t-1]) + np.dot(U, x[t]) + b

#h[t] = tanh(a[t])

#o[t] = np.dot(V, h[t]) + c

#p[t] = softmax(o[t])


