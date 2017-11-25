from .sampler import BaseSampler
from .rejection_sampler import RejectionSampler
from .utils import flatten_function, normalize_vector

import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import time

"""class doc"""
class SMCSampler(BaseSampler):
    # set and get for threshold
    @property
    def thresholds(self):
        return self._thresholds

    @thresholds.setter
    def thresholds(self, thresholds):
        thresholds = np.atleast_1d(thresholds)
        if all(isinstance(t, (int, float)) and (t > 0 or np.isclose(t, 0))  for t in thresholds):
            self._thresholds = thresholds
        else:
            raise ValueError("Passed argument {} must not be a list of integers or float and non-negative".format(thresholds))

    @property
    def particles(self):
        return self._particles

    @property
    def weights(self):
        return self._weights


    def __init__(self, priors, simulator, observation, summaries, distance='euclidean', verbosity=1, seed=None):

        # call BaseSampler __init__
        super().__init__(priors, simulator, observation, summaries, distance, verbosity, seed)

    def sample(self, thresholds, nr_samples, distance='euclidean'):
        """Draw samples using Sequential Monte Carlo.

        Args:
            thresholds: list of acceptance threshold. len(thresholds defines number of SMC iterations)
            nr_particles: Number of particles used to represent the distribution
            distance: distance measure to compare summary statistics. (default) euclidean

        Returns:
            Nothing

        """
        if not thresholds:
            raise ValueError("There must be at least one threshold value.")

        self._thresholds = thresholds
        print("SMC sampler started with thresholds: {} and number of samples: {}".format(self.thresholds, nr_samples))
        self._reset()
        self._run_PMC_sampling(nr_samples)
        if self.verbosity == 1:
            print("Samples: %6d - Thresholds: %.2f - Iterations: %10d - Acceptance rate: %4f - Time: %8.2f s" % (nr_samples, self.thresholds[-1], self.nr_iter, self.acceptance_rate, self.runtime))


    def _calculate_weights(self, curr_theta, prev_thetas, ws, sigma):

        prior_mean = 0

        for i in range(len(curr_theta)):
            prior_mean += self._priors[i].pdf(curr_theta[i])

        prior_mean = prior_mean / len(self._priors)
        kernel = ss.multivariate_normal(curr_theta, sigma, allow_singular=True).pdf
        weight = prior_mean / np.dot(ws, kernel(prev_thetas))

        return weight

    def _run_PMC_sampling(self, nr_samples):
        T = len(self.thresholds)
        X = self.observation

        list_of_stats_x = normalize_vector(flatten_function(self.summaries, X))
        num_priors = len(self.priors) # TODO: multivariate prior?
        nr_iter = 0

        #create a large array to store all particles (THIS CAN BE VERY MEMORY INTENSIVE)
        thetas = np.zeros((T, nr_samples, num_priors))
        weights = np.zeros((T, nr_samples))
        sigma =  np.zeros((T, num_priors, num_priors))
        distances = np.zeros((T, nr_samples))

        start = time.clock()

        for t in range(T):
            #init particles by using ABC Rejection Sampling with first treshold
            if t==0:
                rej_samp = RejectionSampler(
                    priors=self.priors,
                    simulator=self.simulator,
                    summaries=self.summaries,
                    distance=self.distance,
                    observation=self.observation,
                    verbosity = 0
                )
                rej_samp.sample(threshold=self.thresholds[0], nr_samples=nr_samples)

                nr_iter += rej_samp.nr_iter

                thetas[t,:,:] = rej_samp.Thetas
                distances[t,:] = rej_samp.distances
                #create even particle for each
                weights[t,:] = np.ones(nr_samples) / nr_samples
                sigma[t,:,:] = 2*np.cov(thetas[t,:,:].T)
            else:
                print('starting iteration[', t,']')
                for i in range(0, nr_samples):
                    while (True):
                        nr_iter += 1
                        #sample from the previous iteration, with weights and perturb the sample
                        idx = np.random.choice(np.arange(nr_samples), p=weights[t-1,:])
                        theta = np.atleast_1d(thetas[t-1,idx,:])
                        thetap = np.atleast_1d(ss.multivariate_normal(theta,sigma[t-1]).rvs())

                        # for which theta pertubation produced unreasonable values?
                        for id, prior in enumerate(self.priors):
                            if prior.pdf(thetap[id]) == 0:
                                thetap[id] = theta[id]

                        Y = self.simulator(*(np.atleast_1d(thetap)))  # unpack thetas as single arguments for simulator
                        list_of_stats_y = normalize_vector(flatten_function(self.summaries, Y))
                        # either use predefined distance function or user defined discrepancy function
                        d = self.distance(list_of_stats_x, list_of_stats_y)

                        if d <= self.thresholds[t]:
                            distances[t,i] = d
                            thetas[t,i,:] = thetap
                            weights[t,i] = self._calculate_weights(thetas[t,i,:], thetas[t-1,:], weights[t-1,:], sigma[t-1])
                            break

            print('Iteration', t , 'completed')
            weights[t,:] = weights[t,:] / sum(weights[t,:])
            sigma[t,:,:] = 2 * np.cov(thetas[t,:,:].T,aweights=weights[t,:])


        self._runtime = time.clock() - start
        self._nr_iter = nr_iter
        self._acceptance_rate = nr_samples / self.nr_iter
        self._particles = thetas
        self._weights = weights
        self._Thetas = thetas[T-1,:,:]
        self._distances = distances

        return thetas[T-1,:,:]

    def _reset(self):
        """reset class properties for a new call of sample method"""
        self._nr_iter = 0
        self._Thetas = np.empty(0)