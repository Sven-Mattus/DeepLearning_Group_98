import numpy as np
import pickle

class VanillaRNN:
    def __init__(self, sig, m, K):
        self.b = np.zeros((m,1))
        self.c = np.zeros((K,1))
        self.U = np.random.randn(m, K)*sig
        self.W = np.random.randn(m, m)*sig
        self.V = np.random.randn(K, m)*sig

    def save_best_weights(self, loss, loss_val):
        # Check if the loss is smaller than the one saved in loss.txt
        try:
            with open('evaluation/loss_val.txt', 'r') as f:
                saved_loss_val = float(f.read().strip())  # Read the saved loss from the file
        except FileNotFoundError:
            saved_loss_val = float('inf')  # If the file doesn't exist, set saved_loss to infinity

        if loss_val < saved_loss_val:
            
            with open('evaluation/best_weights.pkl', 'wb') as f:
                pickle.dump({
                    'b': self.b,
                    'c': self.c,
                    'U': self.U,
                    'W': self.W,
                    'V': self.V
                }, f)
            with open('evaluation/loss_train.txt', 'w') as f:
                f.write(str(loss))
            with open('evaluation/loss_val.txt', 'w') as f:
                f.write(str(loss_val))


    @classmethod
    def load_weights(cls, filename='evaluation/best_weights.pkl'):
        with open(filename, 'rb') as f:
            weights = pickle.load(f)
        rnn = cls.__new__(cls)
        rnn.b = weights['b']
        rnn.c = weights['c']
        rnn.U = weights['U']
        rnn.W = weights['W']
        rnn.V = weights['V']
        return rnn