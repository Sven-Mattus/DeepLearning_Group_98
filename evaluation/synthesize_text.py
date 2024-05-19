import numpy as np
from math_function import math_functions as mathf

def synthesize_text(RNN, h0, x0, n, data_converter):
    K = RNN.c.shape[0]
    ht = h0
    xt = data_converter.one_hot_encode(x0)
    #xiis = np.zeros(n, 0)
    #Y = np.zeros((K, n))
    output_string = []
    for t in range(n):
        at = np.dot(RNN.W, ht) + np.dot(RNN.U, xt) + RNN.b
        ht = mathf.tanh(at)
        ot = np.dot(RNN.V, ht) + RNN.c
        pt = mathf.softmax(ot)
        
        cp = np.cumsum(pt)
        a = np.random.rand()
        ixs = np.where(cp - a > 0)[0][0]
        ii = ixs

        xt = np.zeros((K,1))
        xt[ii, 0] = 1
    #    xiis[t] = ii
        character = data_converter.one_hot_to_chars(xt)
        #Y[ii, t] = 
        output_string.append(character)
    return output_string

def print_text():
    pass