import numpy as np
from math_function import math_functions as mathf


def compute_gradients_ana(hs, as_, Y, X, P, RNN):
    n = hs.shape[1] - 1
    grad_o = - np.transpose(Y - P)
    grad_h = calculate_gradient_h(hs, grad_o, RNN, as_)
    grad_a = - grad_h * (1 - mathf.tanh(as_) * mathf.tanh(as_)).T
    grad_W = - np.dot((grad_a).T, hs[:, 0:n].T) # TODO is it from 0 on?
    grad_V =  np.dot(grad_o.T, hs[:, 1:n+1].T) #TODO maybe only n
    grad_U = - np.dot(grad_a.T, X.T)
    grad_c = np.sum(grad_o, 0).T 
    grad_b = - np.sum(grad_a, 0).T
    grads = dict('b', grad_b, 'c', grad_c, 'U', grad_U, 'W', grad_W, 'V', grad_V)

    #Implement clipping
    # Iterate over the dictionary and apply np.clip to constrain values
    for key in grads.keys():
        grads[key] = max(min(grads[key], 5), -5)
    return grads
    

def calculate_gradient_h(hs, grad_o, RNN, as_):
    grad_h = np.zeros((grad_o.shape[0], hs.shape[0]))
    T = grad_h.shape[0]
    grad_h[T,:] = np.dot(grad_o[T,:], RNN.V)
    for t in T[::-1]:
        grad_h[T,:] = grad_o[T,:] * RNN.V + np.dot(grad_h[t+1,:] * np.transpose(1 - mathf.tanh(as_(t+1)) * mathf.tanh(as_(t+1))), RNN.W)
    
    return grad_h
