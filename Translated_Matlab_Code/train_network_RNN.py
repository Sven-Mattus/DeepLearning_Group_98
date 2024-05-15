import numpy as np
import ..data_handler 

def TrainNetwork(book_data, nr_iterations, seq_length, RNN, char_to_ind, eta, ind_to_char):
    m = RNN.b.shape[0]
    K = RNN.c.shape[0]
    smooth_losses = nr_iterations.shape[0]
    smooth_loss = 0
    G = dict('b', np.zeros((m,1)), 'c', np.zeros((K,1)), 'U', np.zeros((m, K)), 'W', np.zeros((m, m)), 'V', np.zeros((K, m)))
    e = 0
    hprev = np.zeros((m,1))
    for i in range(nr_iterations):
        X_chars = book_data[e:e+seq_length-1, :]
        Y_chars = book_data[e+1:e+seq_length, :]
        X = charsToOneHot(X_chars, K, char_to_ind)
        Y = charsToOneHot(Y_chars, K, char_to_ind)
        [loss, hs, as_, P] = ForwardPass(hprev, RNN, X, Y)
        if i==0:
            smooth_loss = loss
        
        smooth_loss = .999* smooth_loss + .001 * loss
        grads = ComputeGradsAna(hs, as, Y, X, P, RNN)
        [RNN, G] = AdaGradUpdateStep(RNN, grads, eta, G)
        hprev = hs(:, size(hs, 2)) % last computed hidden state
        smooth_losses(i) = smooth_loss

        if(i % 10000 == 0):
            print(['iter = ', str(i), ', loss = ', str(smooth_loss)])
            print(['iteration ', str(i), ': ', oneHotToChars(SynthesizeText(RNN, np.zeros(m,1), ' ', 200, char_to_ind), ind_to_char)])
        
        e = e+seq_length
        if e + seq_length > len(book_data):
            e = 1
            hprev = np.zeros(m,1)
        
    
    new_RNN = RNN

    return new_RNN, smooth_losses


def AdaGradUpdateStep(RNN, grads, mu, G):
    new_RNN = {}
    for key in RNN.keys:
        G[key] = G[key] + grads[key] * grads[key]
        new_RNN[key] = RNN[key] - (mu/np.sqrt(G[key]) + np.finfo(np.float64).eps) * grads[key]
    
    return new_RNN, G
    
