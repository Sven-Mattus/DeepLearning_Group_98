import numpy as np

class VanillaRNN:
    def __init__(self, sig, m, K):
        self.b = np.zeros((m,1))
        self.c = np.zeros((K,1))
        self.U = np.random.randn(m, K)*sig
        self.W = np.random.randn(m, m)*sig
        self.V = np.random.randn(K, m)*sig