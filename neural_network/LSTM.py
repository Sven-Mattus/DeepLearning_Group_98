
import numpy as np

from math_function.math_functions import sigmoid
from neural_network.INeuralNetwork import INeuralNetwork


class LSTM(INeuralNetwork):

    def __init__(self, input_size, output_size, learning_rate):
        self.W_i, self.W_f, self.W_o, self.W_c, self.U_i, self.U_f, self.U_o, self.U_c = self._init_params(input_size, output_size)
        self.learning_rate = learning_rate

    def train_network(self, dataset, nr_epochs):
        K = self.W_i.shape[0]
        batch_size = dataset[0][0].shape[0]
        hprev = np.zeros([K, batch_size])
        cprev = np.zeros([K, batch_size])
        for t in range(2):  # nr_epochs * len(dataset)):
            (X_t, Y_t) = dataset[t % len(dataset)]
            self._forward_pass(X_t, hprev, cprev)
            # grads = ComputeGradsAna(hs, as, Y, X, P, RNN);
            # [RNN, G] = AdaGradUpdateStep(RNN, grads, eta, G);

    def _init_params(self, input_size, output_size):
        W_f = self._init_weights(input_size, output_size)  # Forget Gate
        U_f = self._init_weights(input_size, output_size)
        W_i = self._init_weights(input_size, output_size)  # Input Gate
        U_i = self._init_weights(input_size, output_size)
        W_c = self._init_weights(input_size, output_size)  # Candidate Gate
        U_c = self._init_weights(input_size, output_size)
        W_o = self._init_weights(input_size, output_size)  # Output Gate
        U_o = self._init_weights(input_size, output_size)
        return W_i, W_f, W_o, W_c, U_i, U_f, U_o, U_c

    def _init_weights(self, input_size, output_size):
        # Xavier Normalized Initialization
        return np.random.uniform(-1, 1, (output_size, input_size)) * np.sqrt(6 / (input_size + output_size))

    def _forward_pass(self, sequence_batch, hprev, cprev):
        """
        evaluates the classifier
        :param sequence_batch: the input data of size BATCH_SIZE x SEQ_LENGTH
        :return: probability vector
        """
        K = self.W_i.shape[0]
        seq_length = sequence_batch.shape[1]
        h_t = hprev  # of size K x BATCH_SIZE
        c_t = cprev  # of size K x BATCH_SIZE
        for i in range(seq_length):
            x = np.asarray(sequence_batch[:, i])
            x_t = self._one_hot_enc(x, K)  # of size K x BATCH_SIZE
            i_t = sigmoid(np.dot(self.W_i, h_t) + np.dot(self.U_i, x_t))  # Input gate of size K x BATCH_SIZE
            f_t = sigmoid(np.dot(self.W_f, h_t) + np.dot(self.U_f, x_t))  # Forget gate of size K x BATCH_SIZE
            o_t = sigmoid(np.dot(self.W_o, h_t) + np.dot(self.U_o, x_t))  # Output / Exposure gate of size K x BATCH_SIZ
            c_wave_t = np.tanh(np.dot(self.W_c, h_t) + np.dot(self.U_c, x_t))  # new memory cell of size K x BATCH_SIZE
            c_t = np.multiply(f_t, c_t) + np.multiply(i_t, c_wave_t)  # Final memory cell of size K x BATCH_SIZE
            h_t = np.multiply(o_t, np.tanh(c_t))

    def _one_hot_enc(self, x, K):
        X = np.zeros([K, len(x)])
        for i in range(len(x)):
            X[x[i], i] = 1
        return X
