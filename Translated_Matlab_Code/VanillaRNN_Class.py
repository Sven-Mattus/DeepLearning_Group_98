import numpy as np

class VanillaRNN:
    def __init__(self, sig, m, K):
        self.b = np.zeroes(m,1)
        self.c = np.zeros(K,1)
        self.U = np.randn(m, K)*sig
        self.W = np.randn(m, m)*sig
        self.V = np.randn(K, m)*sig