from matplotlib import pyplot as plt


class HistoryEvaluator:

    @staticmethod
    def plot_loss(training_history):
        loss = training_history.history['loss']
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(loss, label='Training set')
        plt.legend()
        plt.grid(linestyle='--', linewidth=1, alpha=0.5)
        plt.show()
