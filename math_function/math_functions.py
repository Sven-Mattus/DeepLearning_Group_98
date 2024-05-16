import numpy as np


def sigmoid(t):
    sig = 1 / (1 + np.exp(-t))
    return sig

def sigmoid_derivative(t):
    sig_deriv = sigmoid(t) * (1 - sigmoid(t))
    return sig_deriv

def tanh(x):
    tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return tanh

def tanh_derivative(t):
    tanh_deriv = 1 - tanh(t)*tanh(t)
    return tanh_deriv

def softmax(s):
    softmax = np.exp(s) / sum(np.exp(s))
    return softmax