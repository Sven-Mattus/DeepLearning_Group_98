import numpy as np

def compare_gradients(num_grad, ana_grad):
    for param_name in num_grad.keys():
            num_grad_param = num_grad[param_name]
            ana_grad_param = ana_grad[param_name]
            num_grad_param = num_grad_param.reshape(ana_grad_param.shape)
            difference = num_grad_param - ana_grad_param
            print(f'Gradient difference between num and ana for weight {param_name}:')
            print(difference)