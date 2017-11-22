import numpy as np
import scipy.stats as ss
from functools import partial

SCIPY_ALIASES = {
    'normal': 'norm',
    'gaussian': 'norm',
    'exponential': 'expon',
    'unif': 'uniform',
    'bin': 'binom',
    'binomial': 'binom',
    'multivariate_gaussian': 'multivariate_normal'
}

NUMPY_ALIASES = {
    'normal': 'normal',
    'norm': 'normal',
    'gaussian': 'normal',
    'unif': 'uniform',
    'bin': 'binomial',
    'binom': 'binomial',
    'multivariate_gaussian': 'multivariate_normal',
    'expon': 'exponential'
}

VALID_NUMPY_SAMPLERS = [
    'beta', 'dirichlet', 'f', 'laplace', 'multinomial', 'multivariate_normal',
    'gamma', 'exponential', 'normal', 'uniform', 'binomial', 'poisson'
]


def scipy_from_str(name):
    """Return the scipy.stats distribution corresponding to `name`."""
    name = name.lower()
    name = SCIPY_ALIASES.get(name, name)
    return getattr(ss, name)

def numpy_sampler_from_str(name, *args):
    """Return the numpy.random sampling function corresponding to 'name'."""
    name = name.lower()
    name = NUMPY_ALIASES.get(name, name)
    # only certain distributions work with numpy
    if name in VALID_NUMPY_SAMPLERS:
        np_function = getattr(np.random, name)

        if name == 'uniform':
            # parameters of uniform distribution need to be reshaped
            # from scipy to numpy format
            return partial(np_function, args[0], args[0] + args[1])
        elif name == 'exponential':
            # add the the loc parameter
            return lambda size: np_function(args[1], size=size) + args[0]
        else:
            return partial(np_function, *args)
    else:
        raise ValueError('{} is not a valid numpy.random sampler'.format(name))


def flatten_function(list_of_f, args=None):
    """return function output as 1d array"""
    ret = np.empty(0)
    for f in list_of_f:
        if args is None:
            ret = np.concatenate((ret, np.atleast_1d(f()).flatten()))
        else:
            ret = np.concatenate((ret, np.atleast_1d(f(args)).flatten()))

    return ret
