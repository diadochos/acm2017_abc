import numpy as np
import scipy.stats as ss

SCIPY_ALIASES = {
    'normal': 'norm',
    'gaussian': 'norm',
    'exponential': 'expon',
    'unif': 'uniform',
    'bin': 'binom',
    'binomial': 'binom'
}

NUMPY_ALIASES = {
    'gaussian': 'normal',
    'unif': 'uniform',
    'bin': 'binomial',
}

def scipy_from_str(name):
    """Return the scipy.stats distribution corresponding to `name`."""
    name = name.lower()
    name = SCIPY_ALIASES.get(name, name)
    return getattr(ss, name)

def numpy_sampler_from_str(name):
    """Return the numpy.random sampling function corresponding to 'name'."""
    name = name.lower()
    name = NUMPY_ALIASES.get(name, name)
    return getattr(np.random, name)

def flatten_function(list_of_f, args=None):
    """return function output as 1d array"""
    ret = np.empty(0)
    for f in list_of_f:
        if args is None:
            ret = np.concatenate((ret, np.atleast_1d(f()).flatten()))
        else:
            ret = np.concatenate((ret, np.atleast_1d(f(args)).flatten()))

    return ret
