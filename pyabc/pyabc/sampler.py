#sampler.py

import abc
import sys
import numpy as np
import time


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

    prior = None # callable function used to sample thetas
    simulator = None # callable function used to simulate data, has to be comparable to observation
    observation = None # observed data, can be any numerical numpy array format
    discrepancy = None # own discrepancy function which operates on the data and gives a distance measure: f: R^nxm x R^nxm -> R^1
    summaries = []  # list of summary statistics, each a callabe function of type f: R^nxm -> R^1 (accepts numerical data and give working with simulation and observational data
    nr_iter = [] # list of number of iterations for each sampling process
    thresholds = [] # list of threshholds used to accept or reject samples
    nr_samples = 0 # list of accepted samples per iteration
    Thetas = [] # tensor of drawn model parameters per iteration. Each iteration produces a vector/matrix for each theta_i (ith column)
    verbosity = 0 # level of debug messages

    def set_simulator(self, sim):
        """func doc"""
        if callable(sim) or sim is None:
            self.simulator = sim
        else:
            self._eprint("{}: Passed argument {} is not a callable function!".format(self.__class__, sim))
            sys.exit(1)

    def set_prior(self, prior):
        """func doc"""
        if callable(prior) or prior is None:
            self.prior = prior
        else:
            self._eprint("{}: Passed argument {} is not a callable function!".format(self.__class__, prior))
            sys.exit(1)

    def set_summaries(self, summary):
        """func doc"""
        if callable(summary) or all(isinstance(s, function) for s in summary):
            summary = np.atleast_1d(summary)
            self.summary = summary
        else:
            self._eprint("{}: Passed argument {} is not a callable function or list of functions!".format(self.__class__, summary))
            sys.exit(1)

    def add_summary(self, summary):
        """func doc"""
        if callable(summary):
            self.summary.append(summary)
        else:
            self._eprint("{}: Passed argument {} is not a callable function!".format(self.__class__, summary))
            sys.exit(1)

    def set_discrepancy(self, disc):
        """func doc"""
        if callable(disc) or disc is None:
            self.discrepancy = disc
        else:
            self._eprint("{}: Passed argument {} is not a callable function!".format(self.__class__, disc))
            sys.exit(1)

    def set_observation(self, obs):
        """func doc"""
        if obs is None:
            self.observation = None
        try:
            np.atleast_1d(obs)
            self.observation = obs
        except (TypeError, ValueError, RuntimeError):
            self._eprint("{}: Passed argument {} cannot be parsed by numpy.atleast_1d()!".format(self.__class__, obs))
            sys.exit(1)

    def set_verbosity(self, lvl):
        """set verbosity level of print messages. Possible values are 0, 1, and 2"""
        if type(lvl) == int and lvl >= 0 and lvl <= 2:
            self.verbosity = lvl
        else:
            self._eprint("{}: Passed argument {} has to be integer and between [0,2].".format(self.__class__, lvl))
            sys.exit(1)

    @abc.abstractmethod
    def sample(self):
        pass

    @abc.abstractmethod
    def plot_marginals(self):
        pass

    def _eprint(*args, **kwargs):
        print(*args, file=sys.stderr, **kwargs)


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

    def __init__(self, prior=None, simulator=None, observation=None, discrepancy=None, summaries=[], verbosity=1):
        """constructor"""
        self.set_prior(prior)
        self.set_simulator(simulator)
        self.set_observation(observation)
        self.set_discrepancy(discrepancy)
        self.set_summaries(summaries)
        self.set_verbosity(verbosity)


    def sample(self, thresholds, nr_samples):
        """Main method of sampler. Draw from prior and simulate data until nr_samples were accepted according to threshold.

        Args:
            thresholds: Single value or list of values. For each threshold, the process samples nr_samples thetas. Threshold is used as acceptance criteria.
            nr_samples: Number of samples drawn from prior distribution.

        Returns:
            Nothing

        """
        # check all prerequisites
        thresholds = np.atleast_1d(thresholds)
        if all(isinstance(x, float) for x in thresholds):
            self.thresholds = thresholds
        else:
            self._eprint("{}: Passed argument {} has to be float or a list of floats.".format(self.__class__, thresholds))

        if type(nr_samples) == int:
            self.nr_samples = nr_samples
        else:
            self._eprint("{}: Passed argument {} has to be integer.".format(self.__class__, nr_samples))

        if self.simulator is None or self.prior is None or self.observation is None or len(self.summaries) == 0:
            self._eprint("{}: Method sample() called before all necessary functions are set (prior, simulatior, observation, summaries).".format(self.__class__))

        print("Rejection sampler started with thresholds: {} and number of samples: {}".format(self.thresholds, self.nr_samples))
        run = 0
        for epsilon in self.thresholds:
            X = self.observation
            thetas = np.zeros((self.nr_samples, np.atleast_1d(self.prior()).shape[0]))
            Thetas  = np.zeros((len(self.thresholds), thetas.shape)) # tensor 3rd order: for each epsilon a matrix with nr_samples entries for each theta
            nr_iter = 0
            start = time.clock()

            for i in range(self.nr_samples):
                while True:
                    theta = self.prior()
                    Y = self.simulator(theta)

                    if self.discrepancy is None:
                        stat_vec_x, stat_vec_y = [(s(X), s(Y)) for s in self.summaries]
                        d = np.linalg.norm(stat_vec_x - stat_vec_y)
                        if d < epsilon:
                            thetas[i,:] = theta
                    else:
                        d = self.discrepancy(X,Y)
                        if d < epsilon:
                            thetas[i,:] = theta
                    nr_iter += 1

            end = time.clock()
            self.nr_iter.append(nr_iter)
            Thetas[run] = thetas

            if self.verbosity == 1:
                print("Run: %.2d - Samples: %.6d - Iterations: %.10 - Time: %8f" % (run, self.nr_samples, self.nr_iter[-1], end - start))



    def plot_marginals(self):
        """func doc"""
        pass

"""class doc"""
class SMCSampler(object):
    pass


if __name__ == '__main__':
    print('Subclass:', issubclass(RejectionSampler,
                                  BaseSampler))
    print('Instance:', isinstance(RejectionSampler(),
                                  BaseSampler))
