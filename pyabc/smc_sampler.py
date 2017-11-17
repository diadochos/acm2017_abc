from .sampler import BaseSampler
import matplotlib.pyplot as plt
import numpy as np
import time

"""class doc"""
class SMCSampler(object):
    # set and get for threshold
    @property
    def thresholds(self):
        return self._thresholds

    @thresholds.setter
    def thresholds(self, thresholds):
        thresholds = np.atleast_1d(thresholds)
        if all(isinstance(t, (int, float)) and (t > 0 or np.isclose(t, 0))  for t in thresholds):
            self.thresholds = thresholds
        else:
            raise ValueError("Passed argument {} must not be a list of integers or float and non-negative".format(thresholds))

    def __init__(self, priors, simulator, observation, summaries, discrepancy=None, verbosity=1, seed=None):

        # must have
        self.priors = priors
        self.simulator = simulator
        self.observation = observation
        self.summaries = summaries

        # optional
        self.verbosity = verbosity
        self.discrepancy = discrepancy

        if seed is not None:
            np.random.seed(seed)

    def sample(self, thresholds, nr_particles, distance='euclidean'):
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

        T = len(thresholds)

        for t in range(T):
            pass



    def plot_marginals(self):
        pass
