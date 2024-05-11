# implementation of activation and other mathematical functions

#only used for constans like e or the exp-funct.
import math
import numpy as np

def sigmoid(t):
    sig = 1/(1+(math.e)**(-1))
    return sig

def tanh(x):
    tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return tanh