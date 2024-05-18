import numpy as np
import copy
from Translated_Matlab_Code import forward_pass as fp
from Translated_Matlab_Code import gradient_RNN as gradRNN
from data_handler import DataConverter
from math_function import math_functions as mathf
import numerical_gradients.compute_gradients_numerical as gradnum
import numerical_gradients.compare_gradients as compare


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
        grads_num = gradnum.compute_grad_num_slow(X, Y, RNN, hprev)

        compare.compare_gradients(grads_num, grads)
        
        [RNN, Gradients] = AdaGradUpdateStep(RNN, grads, eta, Gradients)
        hprev = hs[:, hs.shape[1] - 1].reshape(-1, 1)
        smooth_losses.append(smooth_loss)

        if(i % 10000 == 0):
            print(['iter = ', str(i), ', loss = ', str(smooth_loss)])
            synthesized_data_raw = SynthesizeText(RNN, np.zeros((m,1)), ' ', 200, data_converter)
            synthesized_data = data_converter.one_hot_to_chars(synthesized_data_raw)
            print(['iteration ', str(i), ': ', synthesized_data])
        
        e = e+seq_length
        if e + seq_length > len(book_data):
            e = 0
            hprev = np.zeros((m,1))
        
    new_RNN = RNN

    return new_RNN, smooth_losses


def AdaGradUpdateStep(RNN, grads, eta, G):
    new_RNN = copy.deepcopy(RNN)
    for attribute, value in RNN.__dict__.items():
        #beacuse we use classes and dicts here
        key = attribute
        grads_squared = grads[key] * grads[key]
        G[key] = G[key] + grads_squared.reshape(G[key].shape)
        #new_RNN[attribute] = getattr(RNN, attribute) - (eta/np.sqrt(G[key]) + np.finfo(np.float64).eps) * grads[key]   
        new_value = getattr(RNN, attribute) - eta/np.sqrt(G[key] + np.finfo(np.float64).eps) * grads[key].reshape(G[key].shape)
        setattr(new_RNN, attribute, new_value)
        # update = getattr(G, attribute) + getattr(grads, attribute) * getattr(grads, attribute)
        # setattr(G, attribute, update)
        # setattr(new_RNN, attribute, getattr(G, attribute) ) 
    return new_RNN, G
    
def SynthesizeText(RNN, h0, x0, n, data_converter):
    K = RNN.c.shape[0]
    ht = h0
    xt = data_converter.one_hot_encode(x0)
    #xiis = np.zeros(n, 0)
    Y = np.zeros((K, n))
    for t in range(n):
        at = np.dot(RNN.W, ht) + np.dot(RNN.U, xt) + RNN.b
        ht = mathf.tanh(at)
        ot = np.dot(RNN.V, ht) + RNN.c
        pt = mathf.softmax(ot)
        
        cp = np.cumsum(pt)
        a = np.random.rand()
        ixs = np.where(cp - a > 0)[0]
        ii = ixs

        xt = np.zeros((K,0))
        xt[ii] = 1
    #    xiis[t] = ii
        Y[ii, t] = 1

    return Y