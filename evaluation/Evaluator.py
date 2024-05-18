from matplotlib import pyplot as plt

from neural_network.LSTM import LSTM


class Evaluator:

    @staticmethod
    def plot_history_loss(training_history):
        loss = training_history.history['loss']
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(loss, label='Training set')
        plt.legend()
        plt.grid(linestyle='--', linewidth=1, alpha=0.5)
        plt.show()

    @staticmethod
    def plot_validation_loss(self, lstm: LSTM, validation_set):
        pass  # todo


