import numpy as np
import scipy.stats as ss
import pylab as plt
from functools import partial

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

PLOTS_PER_ROW = 3


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


def plot_marginals(sampler):
    """take a sampler and plot the posterior distribution for all model parameter thetas
    :param sampler: instance of BaseSampler
    """
    if sampler.Thetas.shape == (0,):
        raise Warning("Method was called before sampling was done")


    nr_plots = sampler.Thetas.shape[1] # number of columns = model parameters
    nr_rows = (nr_plots // (PLOTS_PER_ROW + 1)) + 1 #has to start by one

    names = np.hstack((np.atleast_1d(p.name) for p in sampler.priors))

    fig = plt.figure()

    for plot_id, thetas in enumerate(sampler.Thetas.T):
        plt.subplot(nr_rows, PLOTS_PER_ROW, plot_id+1)

        plt.hist(thetas, edgecolor="k", bins='auto', normed=True)
        plt.xlabel(names[plot_id])

    try:
        thresholds = getattr(sampler, 'thresholds')
        threshold = thresholds[-1]
    except:
        threshold = getattr(sampler, 'threshold')

    fig.suptitle("Posterior for all model parameters with\n" + r"$\rho(S(X),S(Y)) < {}, n = {}$".format(
        threshold,
        sampler.Thetas.shape[0]
    ), y=1.08)

    plt.tight_layout()
    plt.show()
