import numpy as np

def compare_gradients_absolut(num_grad, ana_grad):
    for param_name in num_grad.keys():
        num_grad_param = num_grad[param_name]
        ana_grad_param = ana_grad[param_name]
        num_grad_param = num_grad_param.reshape(ana_grad_param.shape)
        difference = num_grad_param - ana_grad_param
        max_difference = np.nanmax(np.abs(difference))
        print(f'Max abs gradient difference for weight {param_name}:')
        print(max_difference)


def compare_gradients_relative(num_grad, ana_grad):
    for param_name in num_grad.keys():
        num_grad_param = num_grad[param_name]
        ana_grad_param = ana_grad[param_name]
        num_grad_param = num_grad_param.reshape(ana_grad_param.shape)
        difference = num_grad_param - ana_grad_param
        relative_difference = np.abs(difference) / np.maximum(np.abs(num_grad_param), np.abs(ana_grad_param))
        max_relative_difference = np.nanmax(relative_difference)
        print(f'Max relative gradient difference for {param_name}:')
        print(max_relative_difference)