import numpy as np

def compare_gradients(num_grad, ana_grad):
    for param_name in num_grad.keys():
            num_grad = num_grad[param_name]
            ana_grad = ana_grad[param_name]
            difference = num_grad - ana_grad
            print(f'Gradient difference between num and ana for weight {param_name}:')
            print(difference)