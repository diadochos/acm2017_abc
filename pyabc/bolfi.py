import time
import warnings

import emcee
import numpy as np
import scipy.stats as ss

from .sampler import BaseSampler
from .utils import flatten_function, normalize_vector

from .acquisition import MaxPosteriorVariance
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from GPyOpt.methods import BayesianOptimization


class BOLFI(BaseSampler):
    @property
    def domain(self):
        return self._domain

    @property
    def bolfi(self):
        return self._bolfi

    @domain.setter
    def domain(self, domain):
        if isinstance(domain, list):
            if len(domain) == len(self.priors):
                self._domain = domain
            else:
                raise ValueError('"domain" needs to contain a tuple for every prior in "priors"!')
        else:
            raise TypeError('"domain" needs to be a list!')

    def __init__(self, priors, simulator, observation, summaries, domain, distance='euclidean', verbosity=1, seed=None):

        # call BaseSampler __init__
        super().__init__(priors, simulator, observation, summaries, distance, verbosity, seed)

        self.domain = domain

    def sample(self, nr_samples, threshold, n_chains=2, burn_in=100, **kwargs):
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

        print("BOLFI sampler started with threshold: {} and number of samples: {}".format(self.threshold, nr_samples))
        self._reset()
        self._run_BOLFI_sampling(nr_samples, n_chains, burn_in, **kwargs)

        if self.verbosity == 1:
            print("Samples: %6d - Threshold: keiner - Iterations: %10d - Acceptance rate: %4f - Time: %8.2f s" % (
                nr_samples, self.nr_iter, self.acceptance_rate, self.runtime))

    def _run_BOLFI_sampling(self, nr_samples, n_chains, burn_in, **kwargs):
        # summary statistics of the observed data
        stats_x = normalize_vector(flatten_function(self.summaries, self.observation))

        fill_up, rest = divmod(nr_samples, n_chains)

        # define distance function
        f = lambda thetas: self.distance(stats_x, normalize_vector(
            flatten_function(self.summaries, self.simulator(*thetas.flatten()))))

        # initialize the loop
        accepted_thetas = []
        distances = []

        start = time.clock()

        # create initial evidence set
        # evidence_theta = self.priors.sample(10)
        # evidence_f = np.apply_along_axis(f, axis=1, arr=evidence_theta)

        bounds = [{'name': p.name, 'type': 'continuous', 'domain': domain} for p, domain in
                  zip(self.priors, self.domain)]

        optim = BayesianOptimization(f=f, domain=bounds, acquisition_type='EI',
                                     exact_feval=True, model_type='GP',
                                     num_cores=-1, initial_design_numdata=10,
                                     initial_design_type='sobol')

        max_iter = 30  # evaluation budget
        max_time = 60  # time budget
        eps = 10e-6  # Minimum allows distance between the las two observations

        print("Starting Bayesian Optimization")

        optim.run_optimization(max_iter, max_time, eps)
        self._bolfi = optim

        def loglikelihood(theta, h=0.5):
            # eqn 47 from BOLFI paper
            m, s = optim.model.predict(theta)
            # F = gaussian cdf, see eqn 28 in BOLFI paper
            return ss.norm.logcdf((self.threshold - m) / s).flatten()

        logposterior = lambda theta: loglikelihood(np.atleast_1d(theta)) + self.priors.logpdf(theta)

        # setup EnsembleSampler with nwalkers (chains), dimension of theta vector and a function
        # that returns the natural logarithm of the posterior propability
        sampler = emcee.EnsembleSampler(n_chains, len(self.priors), logposterior)

        # begin mcmc with an exploration phase and store end pos for second run
        p0 = self.priors.sample(n_chains)
        pos = sampler.run_mcmc(p0, burn_in)[0]
        sampler.reset()
        N = divmod(nr_samples, n_chains)[0] + 1
        sampler.run_mcmc(pos, N)

        self._runtime = time.clock() - start

        self._nr_iter = n_chains * N
        self._acceptance_rate = np.mean(sampler.acceptance_fraction)

        # fill thetas equally with samples from all chains
        thetas = sampler.flatchain[:nr_samples]

        self._Thetas = thetas
        # self._distances = distances[:nr_samples]

        return thetas

    def _reset(self):
        """reset class properties for a new call of sample method"""
        self._nr_iter = 0
        self._Thetas = np.empty(0)
