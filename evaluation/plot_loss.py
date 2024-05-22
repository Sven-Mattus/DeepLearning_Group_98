import matplotlib.pyplot as plt

def plot_loss(smooth_loss, smooth_loss_val):    
    plt.plot(smooth_loss)
    plt.plot(smooth_loss_val)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Iteration')
    plt.ylabel('Smooth Loss')
    plt.title('Smooth Loss over Iterations RNN')
    plt.show()
    plt.savefig('evaluation/smooth_loss.png')