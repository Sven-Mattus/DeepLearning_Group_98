import matplotlib.pyplot as plt

def plot_loss(smooth_loss):    
    plt.plot(smooth_loss)
    plt.xlabel('Iteration')
    plt.ylabel('Smooth Loss')
    plt.title('Smooth Loss over Iterations RNN')
    plt.show()
    plt.savefig('evaluation/smooth_loss.png')