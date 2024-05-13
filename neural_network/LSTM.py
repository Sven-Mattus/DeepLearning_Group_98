import numpy as np

from neural_network.INeuralNetwork import INeuralNetwork


class LSTM(INeuralNetwork):

    def __init__(self, input_size, hidden_size, learning_rate):
        self.W_i, self.W_f, self.W_o, self.W_c, self.U_i, self.U_f, self.U_o, self.U_c = self._init_weights(input_size, hidden_size)
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size


    def train_network(self, dataset, nr_epochs):
        h_t = np.zeros()
        c_t = np.zeros()
        for t in range(nr_epochs * len(dataset)):
            x_t = dataset[t][0]
            i_t = np.sigmoid(self.W_i * h_t + self.U_i * x_t)  # Input gate
            f_t = np.sigmoid(self.W_f * h_t + self.U_f * x_t)  # Forget gate
            o_t = np.sigmoid(self.W_o * h_t + self.U_o * x_t)  # Output / Exposure gate
            c_wave_t = np.tanh(self.W_c * h_t + self.U_c * x_t)  # new memory cell
            c_t = np.multiply(f_t, c_t) + np.multiply(i_t, c_wave_t)  # Final memory cell
            h_t = np.multiply(o_t, np.tanh(c_t))

    def init_params(self, input_size, hidden_size):
        W_f = self._init_weights(input_size, hidden_size)  # Forget Gate
        U_f = self._init_weights(input_size, hidden_size)
        W_i = self._init_weights(input_size, hidden_size)  # Input Gate
        U_i = self._init_weights(input_size, hidden_size)
        W_c = self._init_weights(input_size, hidden_size)  # Candidate Gate
        U_c = self._init_weights(input_size, hidden_size)
        W_o = self._init_weights(input_size, hidden_size)  # Output Gate
        U_o = self._init_weights(input_size, hidden_size)
        return W_i, W_f, W_o, W_c, U_i, U_f, U_o, U_c

    def _init_weights(self, input_size, output_size):
        # Xavier Normalized Initialization
        return np.random.uniform(-1, 1, (output_size, input_size)) * np.sqrt(6 / (input_size + output_size))
