import numpy as np

class AttentionHead:

    #initialize attention head instance
    def __init__(self, Q, K, V, batch_size):
        pass

    #compute individual attention
    #based on lecture 11, slide 9
    def layer_normalization(x):
        # Number of elements in x
        d = x.size

        # Compute mean (mu)
        mu = np.sum(x) / d

        # Compute standard deviation (sigma)
        sigma = np.sqrt(np.sum((x - mu) ** 2) / d)