import numpy as np
import forward_pass as fp
import gradient_RNN as gradRNN
from ..data_handler import DataConverter


def TrainNetwork(book_data, nr_iterations, seq_length, RNN, char_to_ind, eta, ind_to_char):
    m = RNN.b.shape[0]
    K = RNN.c.shape[0]
    smooth_losses = []
    G = dict('b', np.zeros((m,1)), 'c', np.zeros((K,1)), 'U', np.zeros((m, K)), 'W', np.zeros((m, m)), 'V', np.zeros((K, m)))
    e = 0
    hprev = np.zeros((m,1))
    data_converter = DataConverter(book_data)
    
    for i in range(nr_iterations):
        X_chars = book_data[e:e+seq_length-1, :]
        Y_chars = book_data[e+1:e+seq_length, :]
        X = charsToOneHot(X_chars, K, char_to_ind)
        Y = charsToOneHot(Y_chars, K, char_to_ind)
        [loss, hs, as_, P] = fp.ForwardPass(hprev, RNN, X, Y)
        if i==0:
            smooth_loss = loss
        
        smooth_loss = .999* smooth_loss + .001 * loss
        grads = gradRNN.compute_gradients_ana(hs, as_, Y, X, P, RNN)
        [RNN, G] = AdaGradUpdateStep(RNN, grads, eta, G)
        hprev = hs[:, hs.shape(1)]
        smooth_losses.append(smooth_loss)

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
    
def SynthesizeText(RNN, h0, x0, n, char_to_ind):
    K = RNN.c.shape 0)
    ht = h0 % size mx1
    xt = charsToOneHot(x0, K, char_to_ind)
    xiis = zeros(n, 0)
    Y = zeros(K, n)
    for t = 1:n
        at = RNN.W * ht + RNN.U * xt + RNN.b
        ht = tanh(at)
        ot = RNN.V * ht + RNN.c
        pt = SoftMax(ot)
        
        % sample next x
        cp = cumsum(pt)
        a = rand
        ixs = find(cp-a >0)
        ii = ixs(1) % index of predicted char

        xt = zeros(K,0)
        xt(ii) = 1
        xiis(t) = ii
        Y(ii, t) = 1

    return Y