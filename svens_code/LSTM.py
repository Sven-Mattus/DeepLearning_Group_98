#Implementation based on equations from Lecture 9, Slides 94. ff.

#use numpys multiply() function for efficient element-wise multiplications of matrices
#np.multiply(array a, array b)
#use np.dot() for "normal" matrix multiplication (dot product)

#class for an LSTM
import numpy as np
from math_functions import *

class LSTM:
    def __init__(self, ):
        pass

    def CreateUnitMatrices(self, m):
        #use numpy hstack to add them columwise
        E_1 = np.hstack([np.identity(m), np.zeros((m, m)), np.zeros((m, m)), np.zeros((m, m))])
        E_2 = np.hstack([np.zeros((m, m))], [np.zeros((m, m))], [np.zeros((m, m))], [np.zeros((m, m))])
        E_3 = np.hstack([np.zeros((m, m))], [np.zeros((m, m))], [np.zeros((m, m))], [np.zeros((m, m))])
        E_4 = np.hstack([np.zeros((m, m))], [np.zeros((m, m))], [np.zeros((m, m))], [np.zeros((m, m))])

        return E_1, E_2, E_3, E_4

    # Input gate
    # i[t] = sigmoid(np.dot(W_i, h[t-1]) + np.dot(U_i, x[t]))

    # # Forget gate
    # f[t] = sigmoid(np.dot(W_f, h[t-1]) + np.dot(U_f, x[t]))

    # # Output gate
    # o[t] = sigmoid(np.dot(W_o, h[t-1]) + np.dot(U_o, x[t]))

    # # New memory cell
    # c_tilde[t] = tanh(np.dot(W_c, h[t-1]) + np.dot(U_c, x[t]))

    # # Final memory cell
    # c[t] = np.multiply(f[t], c[t-1]) + np.multiply(i[t], c[t]ilde[t])

    # # Hidden state
    # h[t] = np.multiply(o[t], tanh(c[t]))

    def ExtracVectors(self, W_all, U_all, h, x, t):

        #get the second dimension of the toal vecotr as m
        a = np.dot(W_all, h[t-1]) + np.dot(U_all, x[t])
        m = a.shape[1]
        #compute the necessary matrices to extract the individual vectors
        E_1, E_2, E_3, E_4 = self.CreateUnitMatrices(m)

        f = sigmoid(E_1, a)
        i = sigmoid(E_2, a)
        o = sigmoid(E_3, a)
        c_tilde = tanh(np.dot(E_4, a))