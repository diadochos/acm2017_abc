import numpy as np

def flatten_function(list_of_f, args=None):
    """return function output as 1d array"""
    ret = np.empty(0)
    for f in list_of_f:
        if args is None:
            ret = np.concatenate((ret, np.atleast_1d(f()).flatten()))
        else:
            ret = np.concatenate((ret, np.atleast_1d(f(args)).flatten()))

    return ret
