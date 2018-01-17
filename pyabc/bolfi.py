import time
import warnings

import emcee
import numpy as np
import scipy.stats as ss

from .sampler import BaseSampler
from .utils import flatten_function, normalize_vector

from .acquisition import MaxPosteriorVariance
import warnings
import GPyOpt


# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     from GPyOpt.methods import BayesianOptimization


class BOLFI(BaseSampler):
    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        if isinstance(domain, list):
            if len(domain) == len(self.priors):
                self._domain = domain
            else:
                raise ValueError('"domain" needs to contain a tuple for every prior in "priors"!')
        else:
            raise TypeError('"domain" needs to be a list!')

    def __init__(self, priors, simulator, observation, summaries, domain, acquisition='LCB', distance='euclidean',
                 verbosity=1, seed=None):

        # call BaseSampler __init__
        super().__init__(priors, simulator, observation, summaries, distance, verbosity, seed)

        self.domain = domain

        # choose the acquisition function as either LCB or MaxVar
        if acquisition.lower() in ['lcb', 'maxvar']:
            self.acqusition_type = acquisition.lower()
        else:
            raise ValueError(
                'acquisition must bei either "lcb" (lower confidence bound) or "maxvar" (maximum posterior variance)')

    def fit(self, initial_evidence_size=10, max_iter=100, **kwargs):
        self._reset()

        # summary statistics of the observed data
        stats_x = flatten_function(self.summaries, self.observation)

        # define distance function
        f = lambda thetas: np.log(self.distance(stats_x,
                                flatten_function(self.summaries, self.simulate(thetas.flatten()))))

        # intialize timer
        start = time.clock()

        # initialize Gaussian Process model
        model = GPyOpt.models.GPModel(verbose=False)

        # create a space
        # TODO: handle discrete parameters differently
        space = GPyOpt.Design_space(space=[{'name': name, 'type': 'continuous', 'domain': domain} for name, domain in
                                           zip(self.priors.names, self.domain)])


        # create GPyOpt object from objective function (distance)
        objective = GPyOpt.core.task.SingleObjective(f)

        # initialize acquisition function
        acquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
        if self.acqusition_type == 'maxvar':
            acquisition = MaxPosteriorVariance(model, space, self.priors, eps=self.threshold, optimizer=acquisition_optimizer)
        elif self.acqusition_type == 'lcb':
            acquisition = GPyOpt.acquisitions.AcquisitionLCB(model, space, optimizer=acquisition_optimizer, exploration_weight=2)

        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

        # initialize by sampling from the prior
        initial_design = self.priors.sample(initial_evidence_size)

        # finally create the Bayesian Optimization object
        self.optim = GPyOpt.methods.ModularBayesianOptimization(model, space, objective, acquisition, evaluator,
                                                           initial_design)

        print("Starting Bayesian Optimization with initial evidence size = {}, max_iter = {}".format(initial_evidence_size, max_iter))

        self.optim.run_optimization(max_iter, **kwargs)

        self._runtime = time.clock() - start


        if self.verbosity == 1:
            print("Fitting BOLFI finished - Time: %8.2f s" % (self.runtime))

    def sample(self, nr_samples, threshold,  n_chains=2, burn_in=100):
        """

        :param threshold:
        :param nr_samples:
        :param n_steps:
        :param burn_in:
        :param kwargs:
        :return:
        """

        if n_chains > nr_samples:
            raise ValueError("number of chains has to be smaller than number of samples")

        self.threshold = threshold

        print("Approximating likelihood with threshold {}".format(self.threshold))

        logposterior = self._logposterior

        # setup EnsembleSampler with nwalkers (chains), dimension of theta vector and a function
        # that returns the natural logarithm of the posterior propability
        sampler = emcee.EnsembleSampler(n_chains, len(self.priors), logposterior)

        print('Obtaining {} MCMC sampling using approximate likelihood with {} chains and {} burn-in samples'.format(nr_samples, n_chains, burn_in))

        # begin mcmc with an exploration phase and store end pos for second run
        p0 = self.priors.sample(n_chains)
        pos = sampler.run_mcmc(p0, burn_in)[0]
        sampler.reset()
        N = divmod(nr_samples, n_chains)[0] + 1
        sampler.run_mcmc(pos, N)


        self._nr_iter = n_chains * N
        self._acceptance_rate = np.mean(sampler.acceptance_fraction)

        if self.verbosity == 1:
            print("Samples: %6d - Acceptance rate: %4f" % (self.nr_iter, self.acceptance_rate))

        # fill thetas equally with samples from all chains
        thetas = sampler.flatchain

        self._Thetas = thetas
        # self._distances = distances[:nr_samples]

        return thetas




    def _loglikelihood(self, theta):
        # eqn 47 from BOLFI paper
        m, v = self.optim.model.predict(theta)
        # F = gaussian cdf, see eqn 28 in BOLFI paper
        return ss.norm.logcdf((np.log(self.threshold) - m) / np.sqrt(v)).flatten()

    # compute posterior from likelihood and prior
    # includes check for bounds
    # TODO: check if log works correctly, when likelihood or prior == 1 -> log = 0
    def _logposterior(self, theta):
        return self._loglikelihood(theta) + self.priors.logpdf(theta)

    def extract_posterior(self, threshold):
        self.threshold = threshold
        return lambda theta: np.exp(self._logposterior(theta))


    def _reset(self):
        """reset class properties for a new call of sample method"""
        self._nr_iter = 0
        self._Thetas = np.empty(0)
        self._simtime = 0

    def log(self):
        """log / save some data"""
        pass
