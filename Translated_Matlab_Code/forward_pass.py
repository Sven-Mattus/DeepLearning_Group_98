import numpy as np
from ..math_function import math_functions as mathf

def ForwardPass(h0, RNN, X, Y):
    ht = h0 # size mx1
    n = X.shape[1]
    m = RNN.b.shape[0]
    K = Y.shape[0]
    
    hs = np.zeros((m,n+1))
    hs[:, 0] = h0
    as_ = np.zeros((m, n))
    loss = 0
    P = np.zeros((K, n))

    for t in range(n):
        at = np.dot(RNN.W, ht) + np.dot(RNN.U, X[:, t]) + RNN.b
        ht = mathf.tanh(at)
        ot = np.dot(RNN.V, ht) + RNN.c
        pt = mathf.SoftMax(ot)
        
        hs[:,t+1] = ht
        as_[:, t] = at
        loss = loss - np.dot(np.transpose(Y[:, t]), np.log(pt))
        P[:,t] = pt

    return loss, hs, as_, P