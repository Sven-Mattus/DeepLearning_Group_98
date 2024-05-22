import numpy as np
from Translated_Matlab_Code.VanillaRNN_Class import VanillaRNN
from evaluation.synthesize_text import synthesize_text, print_synthesized_text
from Translated_Matlab_Code import forward_pass as fp


def evaluate_loaded_weights(SEQ_LENGTH, book_as_ind, data_converter, m):

    RNN_loaded = VanillaRNN.load_weights()
    
    loss_loaded_total = 0
    # e = 133825
    e = 0
    for i in range(50):
        e = e + SEQ_LENGTH * i
        X_loaded = data_converter.one_hot_encode(data_converter.ind_to_chars(book_as_ind[e:e+SEQ_LENGTH]))
        Y_loaded = data_converter.one_hot_encode(data_converter.ind_to_chars(book_as_ind[e+1:e+SEQ_LENGTH+1]))
        loss_loaded, _, _, _= fp.ForwardPass(np.zeros((m,1)), RNN_loaded, X_loaded, Y_loaded)
        loss_loaded_total += loss_loaded
    loss_loaded = loss_loaded_total/50

    text = synthesize_text(RNN_loaded, np.zeros((m,1)), ' ', 1000, data_converter)
    print_synthesized_text(text)
