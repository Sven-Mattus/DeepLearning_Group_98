import numpy as np
from Translated_Matlab_Code import forward_pass as fp
from Translated_Matlab_Code import gradient_RNN as gradRNN
from data_handler import DataConverter
from math_function import math_functions as mathf


def TrainNetwork(book_data, nr_iterations, seq_length, RNN,  eta, data_converter):
    m = RNN.b.shape[0]
    K = RNN.c.shape[0]
    smooth_losses = []
    Gradients = {'b': np.zeros((m,1)), 'c': np.zeros((K,1)), 'U': np.zeros((m, K)), 'W': np.zeros((m, m)), 'V': np.zeros((K, m))}
    e = 0
    hprev = np.zeros((m,1))
    
    for i in range(nr_iterations):
        X_chars = book_data[e:e+seq_length]
        Y_chars = book_data[e+1:e+seq_length+1]
        X = data_converter.one_hot_encode(X_chars)
        Y = data_converter.one_hot_encode(Y_chars)
        [loss, hs, as_, P] = fp.ForwardPass(hprev, RNN, X, Y)
        if i==0:
            smooth_loss = loss
        
        smooth_loss = .999* smooth_loss + .001 * loss
        grads = gradRNN.compute_gradients_ana(hs, as_, Y, X, P, RNN)
        [RNN, Gradients] = AdaGradUpdateStep(RNN, grads, eta, Gradients)
        hprev = hs[:, hs.shape(1)]
        smooth_losses.append(smooth_loss)

        if(i % 10000 == 0):
            print(['iter = ', str(i), ', loss = ', str(smooth_loss)])
            synthesized_data_raw = SynthesizeText(RNN, np.zeros(m,1), ' ', 200, data_converter)
            synthesized_data = data_converter.one_hot_to_chars(synthesized_data_raw)
            print(['iteration ', str(i), ': ', synthesized_data])
        
        e = e+seq_length
        if e + seq_length > len(book_data):
            e = 0
            hprev = np.zeros((m,1))
        
    new_RNN = RNN

    return new_RNN, smooth_losses


def AdaGradUpdateStep(RNN, grads, eta, G):
    new_RNN = {}
    for key in RNN.keys():
        G[key] = G[key] + grads[key] * grads[key]
        new_RNN[key] = RNN[key] - (eta/np.sqrt(G[key]) + np.finfo(np.float64).eps) * grads[key]
    
    return new_RNN, G
    
def SynthesizeText(RNN, h0, x0, n, data_converter):
    K = RNN.c.shape[0]
    ht = h0
    xt = data_converter.one_hot_encode(x0)
    #xiis = np.zeros(n, 0)
    Y = np.zeros(K, n)
    for t in range(n+1):
        at = RNN.W * ht + RNN.U * xt + RNN.b
        ht = mathf.tanh(at)
        ot = RNN.V * ht + RNN.c
        pt = mathf.SoftMax(ot)
        
        cp = mathf.cumsum(pt)
        a = np.rand
        ixs = np.where(cp - a > 0)[0]
        ii = ixs[0]

        xt = np.zeros(K,0)
        xt[ii] = 1
    #    xiis[t] = ii
        Y[ii, t] = 1

    return Y