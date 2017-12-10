from functools import partial

import numpy as np
import scipy.stats as ss

# some aliases for convenient call of scipy functions
SCIPY_ALIASES = {
    'normal': 'norm',
    'gaussian': 'norm',
    'exponential': 'expon',
    'unif': 'uniform',
    'bin': 'binom',
    'binomial': 'binom',
    'multivariate_gaussian': 'multivariate_normal'
}

# aliases to get numpy function from scipy name
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

# some samplers from numpy do not work due to differences in parametrization from scipy
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
            # add the loc parameter
            return lambda size: np_function(args[1], size=size) + args[0]
        elif name == 'gamma':
            return lambda size: np_function(args[0], args[2], size=size) + args[1]
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


def normalize_vector(v):
    """normalize vector so that maximum value has value of 1"""
    v_norm = np.linalg.norm(v)
    if v_norm:
        v = v / v_norm

    return v


def numgrad(fn, x, h=None, replace_neg_inf=True):
    """Naive numeric gradient implementation for scalar valued functions.
    Parameters
    ----------
    fn
    x : np.ndarray
        A single point in 1d vector
    h : float or list
        Stepsize or stepsizes for the dimensions
    replace_neg_inf : bool
        Replace neg inf fn values with gradient 0 (useful for logpdf gradients)
    Returns
    -------
    grad : np.ndarray
        1D gradient vector
    """
    h = 0.00001 if h is None else h
    h = np.asanyarray(h).reshape(-1)

    x = np.asanyarray(x, dtype=np.float).reshape(-1)
    dim = len(x)
    X = np.zeros((dim * 3, dim))

    for i in range(3):
        Xi = np.tile(x, (dim, 1))
        np.fill_diagonal(Xi, Xi.diagonal() + (i - 1) * h)
        X[i * dim:(i + 1) * dim, :] = Xi

    f = np.apply_along_axis(fn, axis=1, arr=X)
    # TODO: batch applied logpdf for this: f = fn(X)
    f = f.reshape((3, dim))

    if replace_neg_inf:
        if np.any(np.isneginf(f)):
            return np.zeros(dim)

    grad = np.gradient(f, *h, axis=0)
    return grad[1, :]
