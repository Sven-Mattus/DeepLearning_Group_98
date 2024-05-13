import numpy as np


def sigmoid(t):
    sig = 1 / (1 + np.exp(-t))
    return sig


def tanh(x):
    tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return tanh
