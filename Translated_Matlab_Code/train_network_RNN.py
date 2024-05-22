import numpy as np
import copy
from Translated_Matlab_Code import forward_pass as fp
from Translated_Matlab_Code import gradient_RNN as gradRNN
import evaluation.compute_gradients_numerical as gradnum
import evaluation.compare_gradients as compare
import evaluation.synthesize_text as synthesize


def TrainNetwork(book_data, nr_iterations, seq_length, RNN,  eta, data_converter):
    m = RNN.b.shape[0]
    K = RNN.c.shape[0]
    smooth_losses = []
    Gradients = {'b': np.zeros((m,1)), 'c': np.zeros((K,1)), 'U': np.zeros((m, K)), 'W': np.zeros((m, m)), 'V': np.zeros((K, m))}
    e = 0
    hprev = np.zeros((m,1))
    epoch = 0
    best_loss = 1000
    
    for i in range(nr_iterations):
        X_chars = book_data[e:e+seq_length]
        Y_chars = book_data[e+1:e+seq_length+1]
        X = data_converter.one_hot_encode(X_chars)
        Y = data_converter.one_hot_encode(Y_chars)
        [loss, hs, as_, P] = fp.ForwardPass(hprev, RNN, X, Y)
        if i==0:
            smooth_loss = loss
        
        smooth_loss = .999* smooth_loss + .001 * loss
        grads = gradRNN.compute_gradients_ana(hs, as_, Y, X, P, RNN)
        #grads_num = gradnum.compute_gradients_num(X, Y, RNN, hprev)

        #compare.compare_gradients_absolut(grads_num, grads)
        #compare.compare_gradients_relative(grads_num, grads)
        
        RNN = AdaGradUpdateStep(RNN, grads, eta, Gradients)
        hprev = hs[:, hs.shape[1] - 1].reshape(-1, 1)
        smooth_losses.append(smooth_loss)



        if(i % 10000 == 0 and i < 100000):
            #print(['iter = ', str(i), ', loss = ', str(smooth_loss)])
            synthesized_data = synthesize.synthesize_text(RNN, hprev, X_chars[0], 200, data_converter)

            print(['Synthesized text at teration ', str(i), ': ', synthesized_data])
            #print('Synthesized Text:', end=' ')
            for char in synthesized_data:
                print(char[0], end='')
            print('')
        
        #if(i % 1000 == 0):
            #print('iteration: ', i, 'smooth_loss:', smooth_loss)

        e = e+seq_length
        if e + seq_length > len(book_data):
            e = 0
            hprev = np.zeros((m,1))
            epoch += 1
            print('epoch:', epoch, 'smooth_loss:', smooth_loss)

        if smooth_loss < best_loss:
            best_loss = smooth_loss
            RNN.save_best_weights(best_loss)

        
    new_RNN = RNN

    return new_RNN, smooth_losses


def AdaGradUpdateStep(RNN, grads, eta, G):
    new_RNN = copy.deepcopy(RNN)
    for attribute, value in RNN.__dict__.items():
        #beacuse we use classes and dicts here
        key = attribute
        grads_squared = grads[key] * grads[key]
        G[key] = G[key] + grads_squared.reshape(G[key].shape)
        #new_RNN[attribute] = getattr(RNN, attribute) - (eta/np.sqrt(G[key]) + np.finfo(np.float64).eps) * grads[key]   
        new_value = getattr(RNN, attribute) - eta/np.sqrt(G[key] + np.finfo(np.float64).eps) * grads[key].reshape(G[key].shape)
        setattr(new_RNN, attribute, new_value)
        # update = getattr(G, attribute) + getattr(grads, attribute) * getattr(grads, attribute)
        # setattr(G, attribute, update)
        # setattr(new_RNN, attribute, getattr(G, attribute) ) 
    return new_RNN
    
