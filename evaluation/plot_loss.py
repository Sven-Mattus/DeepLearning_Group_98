def plot_loss(smooth_loss):
    import matplotlib.pyplot as plt
    plt.plot(smooth_loss)
    plt.xlabel('Iteration')
    plt.ylabel('Smooth Loss')
    plt.title('Smooth Loss over Iterations')
    plt.show()