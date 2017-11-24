import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
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


def plot_marginals(sampler, kde=False, **kwargs):
    """take a sampler and plot the posterior distribution for all model parameter thetas
    :param sampler: instance of BaseSampler
    """
    if sampler.Thetas.shape == (0,):
        raise Warning("Method was called before sampling was done")


    nr_plots = sampler.Thetas.shape[1] # number of columns = model parameters
    nr_rows = (nr_plots // (PLOTS_PER_ROW + 1)) + 1 #has to start by one

    names = np.hstack((np.atleast_1d(p.name) for p in sampler.priors))

    fig = plt.figure()

    # plot thetas of last iteration
    for plot_id, thetas in enumerate(sampler.Thetas.T):
        plt.subplot(nr_rows, PLOTS_PER_ROW, plot_id+1)

        # plot posterior
        plt.hist(thetas, edgecolor="k", bins='auto', normed=True, alpha=0.4)
        # plot mean
        plt.axvline(np.mean(thetas), linewidth=1.2, color="m", linestyle="--", label="mean")
        # plot MAP
        if kde:
            # get the bandwidth method argument for scipy
            # and run scipy's kde
            kde = ss.kde.gaussian_kde(thetas, bw_method=kwargs.get('bw_method'))
            xx = np.linspace(np.min(thetas)-0.1, np.max(thetas)+0.1, 200)
            dens = kde(xx)
            plt.plot(xx, dens)
            plt.axvline(xx[np.argmax(dens)],linewidth=1.2, color="m", linestyle=":", label="MAP")


        # label of axis
        plt.xlabel(names[plot_id])
        plt.legend(loc="upper right")


    # generate title
    try:
        thresholds = getattr(sampler, 'thresholds')
        threshold = thresholds[-1]
    except:
        threshold = getattr(sampler, 'threshold')

    fig.suptitle("Posterior for all model parameters with\n" + r"$\rho(S(X),S(Y)) < {}, n = {}$".format(
        threshold,
        sampler.Thetas.shape[0]
    ), y=0.96)

    plt.tight_layout(rect=[0.05,0,0.95,0.85])
    plt.show()
