#sampler.py

import abc
import sys
import numpy as np
import time
import pylab as plt
import warnings


class BaseSampler(metaclass=abc.ABCMeta):
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    # single value parameters
    simulator = None  # callable function used to simulate data, has to be comparable to observation
    discrepancy = None  # own discrepancy function which uses the summary functions to compute a distance measure: f: R^nxm x R^nxm x ...-> R^1
    nr_iter = 0  # list of number of iterations for each sampling process
    threshold = 0  # list of threshholds used to accept or reject samples
    nr_samples = 0  # list of accepted samples per iteration
    verbosity = 0  # level of debug messages
    runtime = 0 # runtime of abc rejection sampler

    # list value parameters
    priors = [] # list of callable functions used to sample thetas
    summaries = []  # list of summary statistics, each a callabe function of type f: R^nxm -> R^1 (accepts numerical data and give working with simulation and observational data

    # dict value parameters
    distances = {
        'euclidean': lambda x,y: np.linalg.norm(x-y)
    }

    # array value parameters
    observation = np.empty(0) # observed data, can be any numerical numpy array format R^nxm
    Thetas = np.empty(0)  # matrix with all accepted samples for each model parameter. One column represents all samples of one parameter

    def set_simulator(self, sim):
        """func doc"""
        if callable(sim):
            self.simulator = sim
        else:
            raise TypeError("Passed argument {} is not a callable function!".format(sim))

    def set_priors(self, priors):
        """func doc"""
        priors = np.atleast_1d(priors)
        if all(callable(p) for p in priors):
            self.priors = priors
        else:
            raise TypeError("Passed argument {} is not a callable function or list of functions!".format(priors))

    def set_summaries(self, summaries):
        """func doc"""
        summaries = np.atleast_1d(summaries)
        if all(callable(s) for s in summaries):
            self.summaries = summaries
        else:
            raise TypeError("Passed argument {} is not a callable function or list of functions!".format(summaries))

    def add_summary(self, summary):
        """func doc"""
        if callable(summary):
            self.summaries = np.hstack((self.summaries, summary))
        else:
            raise TypeError("Passed argument {} is not a callable function!".format(summary))

    def add_prior(self, prior):
        """func doc"""
        if callable(prior):
            self.priors = np.hstack((self.priors, prior))
        else:
            raise TypeError("Passed argument {} is not a callable function!".format(prior))

    def set_discrepancy(self, disc):
        """func doc"""
        if callable(disc):
            self.discrepancy = disc
        else:
            raise TypeError("Passed argument {} is not a callable function!".format(disc))

    def set_observation(self, obs):
        """func doc"""
        try:
            obs = np.atleast_1d(obs)
            self.observation = obs
        except (TypeError):
            raise TypeError("Passed argument {} cannot be parsed by numpy.atleast_1d()!".format(obs))

    def set_verbosity(self, lvl):
        """set verbosity level of print messages. Possible values are 0, 1, and 2"""
        if isinstance(lvl, int):
            if lvl >= 0 and lvl <= 2:
                self.verbosity = lvl
            else:
                raise ValueError("Passed argument {} has to be one of {0,1,2}.".format(lvl))
        else:
            raise TypeError("Passed argument {} has to be an integer".format(lvl))

    @abc.abstractmethod
    def sample(self):
        pass

    @abc.abstractmethod
    def plot_marginals(self):
        pass

    @abc.abstractmethod
    def _are_you_fucking_ready(self):
        pass

    @abc.abstractmethod
    def _reset(self):
        pass


class RejectionSampler(BaseSampler):
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self, priors=None, simulator=None, observation=None, discrepancy=None, summaries=None, verbosity=1):
        """constructor"""
        if priors is not None:
            self.set_priors(priors)
        if simulator is not None:
            self.set_simulator(simulator)
        if observation is not None:
            self.set_observation(observation)
        if discrepancy is not None:
            self.set_discrepancy(discrepancy)
        if summaries is not None:
            self.set_summaries(summaries)

        self.set_verbosity(verbosity)

    def _are_you_fucking_ready(self):
        """check all prerequisites for the sample method to run through"""
        return self.simulator is not None and len(self.priors) > 0 and len(self.summaries) > 0 and self.observation.shape != (0,)

    def _reset(self):
        """reset class properties for a new call of sample method"""
        self.nr_iter = []
        self.Thetas = []

    def sample_from_priors(self):
        """draw samples from all priors and return as 1d array"""
        return self._flatten_function(self.priors)

    def _flatten_function(self, list_of_f, args=None):
        """return function output as 1d array"""
        ret = np.empty(0)
        for f in list_of_f:
            if args is None:
                ret = np.concatenate((ret, np.atleast_1d(f())))
            else:
                ret = np.concatenate((ret, np.atleast_1d(f(args))))

        return ret


    def _run_rejection_sampling(self, distance):
        """the abc rejection sampling algorithm"""

        X = self.observation

        stat_vec_x = self._flatten_function(self.summaries, args=X)

        thetas = np.zeros((self.nr_samples, self.sample_from_priors().shape[0]))

        nr_iter = 0
        start = time.clock()
        for i in range(self.nr_samples):
            while True:
                nr_iter += 1

                thetas_prop = self.sample_from_priors()  # draw as many thetas as there are priors
                Y = self.simulator(*thetas_prop)  # unpack thetas as single arguments for simulator
                stat_vec_y = self._flatten_function(self.summaries, args=Y)

                # either use predefined distance function or user defined discrepancy function
                if self.discrepancy is None:
                    d = self.distances[distance](stat_vec_x, stat_vec_y)
                else:
                    d = self.discrepancy(stat_vec_x, stat_vec_y)

                if d < self.threshold:
                    thetas[i, :] = thetas_prop
                    break

        self.runtime = time.clock() - start

        self.nr_iter = nr_iter
        self.Thetas = thetas

    def sample(self, threshold, nr_samples, distance='euclidean'):
        """Main method of sampler. Draw from prior and simulate data until nr_samples were accepted according to threshold.

        Args:
            threshold: Threshold is used as acceptance criteria for samples.
            nr_samples: Number of samples drawn from prior distribution.
            distance: distance measure to compare summary statistics. (default) euclidean

        Returns:
            Nothing

        """
        # check all prerequisites
        if isinstance(threshold, (int, float)):
            if threshold > 0 or np.isclose(threshold, 0):
                self.threshold = threshold
            else:
                raise ValueError("Passed argument {} must not be negative".format(threshold))
        else:
            raise TypeError("Passed argument {} has to be and integer or float.".format(threshold))

        if isinstance(nr_samples, (int,float)):
            if nr_samples > 0:
                self.nr_samples = int(nr_samples)
            else:
                raise ValueError("Passed argument {} must not be negative".format(nr_samples))
        else:
            raise TypeError("Passed argument {} has to be integer.".format(nr_samples))

        if distance not in self.distances.keys():
            raise ValueError("Passed distance function {} is not available. Possible values are {}.".format(distance, self.distances.keys()))

        if not self._are_you_fucking_ready():
            raise Warning("At least one necessary function or the observations are not yet set (prior, simulatior, summaries or observation).")

        print("Rejection sampler started with threshold: {} and number of samples: {}".format(self.threshold, self.nr_samples))

        self._reset()

        # RUN ABC REJECTION SAMPLING
        self._run_rejection_sampling(distance)

        if self.verbosity == 1:
            print("Samples: %6d - Threshold: %.2f - Iterations: %10d - Time: %8.2f s" % (self.nr_samples, self.threshold, self.nr_iter, self.runtime))

    def plot_marginals(self, names=[]):
        """func doc"""
        if self.Thetas.shape == (0,):
            raise Warning("Method was called before sampling was done")

        nr_plots = self.Thetas.shape[1] # number of columns

        fig, ax = plt.subplots(1, nr_plots)

        for plot_id, hist in enumerate(self.Thetas.T):
            _ax = ax[plot_id]
            _ax.hist(hist, edgecolor="k", bins='auto', normed=True)
            if names and len(names) == nr_plots:
               _ax.set_xlabel(names[plot_id])

        fig.suptitle("Posterior for all model parameters with\n" + r"$\rho(S(X),S(Y)) < {}, n = {}$".format(self.threshold, self.nr_samples))
        plt.show()

    def __str__(self):
        return "{} - priors: {} - simulator: {} - summaries: {} - observation: {} - discrepancy: {} - verbosity: {}".format(
            type(self).__name__, len(self.priors), self.simulator, len(self.summaries), self.observation.shape, self.discrepancy, self.verbosity
        )

"""class doc"""
class SMCSampler(object):

    def __init__(self, priors=None, simulator=None, observation=None, discrepancy=None, summaries=None, verbosity=1):
        if priors is not None:
            self.set_priors(priors)
        if simulator is not None:
            self.set_simulator(simulator)
        if observation is not None:
            self.set_observation(observation)
        if discrepancy is not None:
            self.set_discrepancy(discrepancy)
        if summaries is not None:
            self.set_summaries(summaries)

        self.set_verbosity(verbosity)

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



    def plot_marginals():
        pass




if __name__ == '__main__':
    print('Subclass:', issubclass(RejectionSampler,
                                  BaseSampler))
    print('Instance:', isinstance(RejectionSampler(),
                                  BaseSampler))
