#sampler.py

import abc
import sys
import numpy as np
import time
import pylab as plt
import warnings


class BaseSampler(metaclass=abc.ABCMeta):
    """Abstract base class for all samplers. Defines common setters and properties

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

    # dict value parameters
    __distances = {
        'euclidean':
            lambda ls1, ls2: np.linalg.norm(np.array([np.linalg.norm(x-y) for x,y in zip(ls1,ls2)]))
    }

    # set and get simulator
    @property
    def simulator(self):
        return self._simulator

    @simulator.setter
    def simulator(self, sim):
        """func doc"""
        if callable(sim):
            self._simulator = sim
        else:
            raise TypeError("Passed argument {} is not a callable function!".format(sim))

    # set and get priors
    @property
    def priors(self):
        return self._priors

    @priors.setter
    def priors(self, priors):
        """func doc"""
        priors = np.atleast_1d(priors)
        if all(callable(p) for p in priors):
            self._priors = priors
        else:
            raise TypeError("Passed argument {} is not a callable function or list of functions!".format(priors))

    # set and get summaries
    @property
    def summaries(self):
        return self._summaries

    @summaries.setter
    def summaries(self, summaries):
        """func doc"""
        summaries = np.atleast_1d(summaries)
        if all(callable(s) for s in summaries):
            self._summaries = summaries
        else:
            raise TypeError("Passed argument {} is not a callable function or list of functions!".format(summaries))

    # set and get discrepancy
    @property
    def discrepancy(self):
        return self._discrepancy

    @discrepancy.setter
    def discrepancy(self, disc):
        """func doc"""
        if callable(disc) or disc is None:
            self._discrepancy = disc
        else:
            raise TypeError("Passed argument {} is not a callable function!".format(disc))

    # set and get observation
    @property
    def observation(self):
        return self._observation

    @observation.setter
    def observation(self, obs):
        """func doc"""
        try:
            obs = np.atleast_1d(obs)
            self._observation = obs
        except (TypeError):
            raise TypeError("Passed argument {} cannot be parsed by numpy.atleast_1d()!".format(obs))

    # set and get verbosity
    @property
    def verbosity(self):
        return self._verbosity

    @verbosity.setter
    def verbosity(self, lvl):
        """set verbosity level of print messages. Possible values are 0, 1, and 2"""
        if isinstance(lvl, int):
            if lvl >= 0 and lvl <= 2:
                self._verbosity = lvl
            else:
                raise ValueError("Passed argument {} has to be one of {0,1,2}.".format(lvl))
        else:
            raise TypeError("Passed argument {} has to be an integer".format(lvl))

    # set and get nr_samples
    @property
    def nr_samples(self):
        return self._nr_samples

    @nr_samples.setter
    def nr_samples(self, nr_samples):
        if isinstance(nr_samples, (int,float)):
            if nr_samples > 0:
                self._nr_samples = int(nr_samples)
            else:
                raise ValueError("Passed argument {} must not be negative".format(nr_samples))
        else:
            raise TypeError("Passed argument {} has to be integer.".format(nr_samples))

    # only getter
    @property
    def nr_iter(self):
        return self._nr_iter

    @property
    def Thetas(self):
        return self._Thetas

    @property
    def runtime(self):
        return self._runtime

    @property
    def distances(self):
        return self.__distances


    @abc.abstractmethod
    def sample(self):
        pass

    @abc.abstractmethod
    def plot_marginals(self):
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
    # set and get for threshold
    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        if isinstance(threshold, (int, float)):
            if threshold > 0 or np.isclose(threshold, 0):
                self._threshold = threshold
            else:
                raise ValueError("Passed argument {} must not be negative".format(threshold))
        else:
            raise TypeError("Passed argument {} has to be and integer or float.".format(threshold))

    def __init__(self, priors, simulator, observation, summaries, discrepancy=None, verbosity=1, seed=None):
        """constructor"""
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

    def _reset(self):
        """reset class properties for a new call of sample method"""
        self._nr_iter = 0
        self._Thetas = np.empty(0)

    def sample_from_priors(self):
        """draw samples from all priors and return as list of outputs

        :return list of outputs for each prior
        """
        return [p() for p in self.priors]

    def _flatten_output(self, list):
        """return function output as 1d array"""
        ret = np.empty(0)
        for e in list:
            ret = np.concatenate((ret, np.atleast_1d(e.flatten())))

        return ret

    def _run_rejection_sampling(self, distance):
        """the abc rejection sampling algorithm"""

        X = self.observation

        list_of_stats_x = [s(X) for s in self.summaries]

        thetas = np.zeros((self.nr_samples, sum(a.shape[0] for a in self.sample_from_priors())))

        nr_iter = 0
        start = time.clock()
        for i in range(self.nr_samples):
            while True:
                nr_iter += 1

                thetas_prop = self.sample_from_priors()  # draw as many thetas as there are priors
                Y = self.simulator(*thetas_prop)  # unpack thetas as single arguments for simulator
                list_of_stats_y = [s(Y) for s in self.summaries]

                if any(s1.shape != s2.shape for s1,s2 in zip(list_of_stats_x, list_of_stats_y)):
                    raise ValueError("Dimensions of summary statistics for observation X ({}) and simulation data Y ({}) are not the same".format(stat_vec_x.shape, stat_vec_y.shape))

                # either use predefined distance function or user defined discrepancy function
                if self.discrepancy is None:
                    d = self.distances[distance](list_of_stats_x, list_of_stats_y)
                else:
                    d = self.discrepancy(*list_of_stats_x, *list_of_stats_y)

                if d < self.threshold:
                    thetas[i, :] = self._flatten_output(thetas_prop)
                    break

        self._runtime = time.clock() - start

        self._nr_iter = nr_iter
        self._Thetas = thetas

    def sample(self, threshold, nr_samples, distance='euclidean'):
        """Main method of sampler. Draw from prior and simulate data until nr_samples were accepted according to threshold.

        Args:
            threshold: Threshold is used as acceptance criteria for samples.
            nr_samples: Number of samples drawn from prior distribution.
            distance: distance measure to compare summary statistics. (default) euclidean

        Returns:
            Nothing

        """
        self.threshold = threshold
        self.nr_samples = nr_samples

        # check prerequisites
        if distance not in self.distances.keys():
            raise ValueError("Passed distance function {} is not available. Possible values are {}.".format(distance, self.__distances.keys()))

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




if __name__ == '__main__':
    print('Subclass:', issubclass(RejectionSampler,
                                  BaseSampler))
    print('Instance:', isinstance(RejectionSampler(),
                                  BaseSampler))
