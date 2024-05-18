import numpy as np
import Translated_Matlab_Code.forward_pass as fp
import copy

def ComputeGradsNum(X, Y, RNN, h):
    num_grads = {}
    for f in vars(RNN):
        print('Computing numerical gradient for')
        print('Field name:', f)
        num_grads[f] = ComputeGradNumSlow(X, Y, f, RNN, h)
    return num_grads

"""def ComputeGradNumSlow(X, Y, f, RNN, h):
    n = np.prod(getattr(RNN, f).shape)
    grad = np.zeros_like(getattr(RNN, f))
    hprev = np.zeros((getattr(RNN, 'W').shape[0], 1))
    for i in range(n):
        RNN_try = copy.deepcopy(RNN)
        rnn_new = getattr(RNN_try, f) - h
        setattr(RNN_try, f, rnn_new)
        l1, _, _, _ = fp.ForwardPass(hprev, RNN_try, X, Y)
        rnn_new = getattr(RNN_try, f) + 2 * h
        l2, _, _, _ = fp.ForwardPass(hprev, RNN_try, X, Y)
        grad.flat[i] = (l2 - l1) / (2 * h)
    return grad
"""
def compute_grad_num_slow(X, Y, param_name, RNN, hprev):
    step_size = 0.0001
    # Get the parameter as a numpy array
    param = getattr(RNN, param_name)
    n = param.size
    grad = np.zeros(param.shape)

    # It's useful to flatten and reshape for generic parameter dimensions
    param_flat = param.flatten()
    for i in range(n):
        RNN_try = copy.deepcopy(RNN)

        # Perturb the parameter down
        param_flat[i] -= step_size
        setattr(RNN_try, param_name, param_flat.reshape(param.shape))
        l1, _, _, _ = fp.ForwardPass(hprev, RNN_try, X, Y)

        # Perturb the parameter up
        param_flat[i] += 2 * step_size
        setattr(RNN_try, param_name, param_flat.reshape(param.shape))
        l2, _, _, _ = fp.ForwardPass(hprev, RNN_try, X, Y)

        # Reset the parameter
        param_flat[i] -= step_size
        idx = np.unravel_index(i, param.shape)
        grad[idx] = (l2 - l1) / (2 * step_size)
    return grad