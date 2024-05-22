import numpy as np

class AttentionHead:

    #initialize attention head instance
    def __init__(self, Q, K, V, batch_size):


    #compute individual attention
    #based on lecture 11, slide 9
    def individual_attention():
    scores = np.matmul(KX.T, QX) / np.sqrt(d)
    attention_weights = softmax(scores)
    
    # Calculate the output
    O = np.matmul(VX, attention_weights)
    return O, attention_weights