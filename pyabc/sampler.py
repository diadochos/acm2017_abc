#sampler.py

import abc
import sys
import numpy as np
import time
import pylab as plt


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

    priors = [] # list of callable functions used to sample thetas
    simulator = None # callable function used to simulate data, has to be comparable to observation
    observation = np.empty(0) # observed data, can be any numerical numpy array format R^nxm
    discrepancy = None # own discrepancy function which operates on the data and gives a distance measure: f: R^nxm x R^nxm -> R^1
    summaries = []  # list of summary statistics, each a callabe function of type f: R^nxm -> R^1 (accepts numerical data and give working with simulation and observational data
    nr_iter = [] # list of number of iterations for each sampling process
    thresholds = [] # list of threshholds used to accept or reject samples
    nr_samples = 0 # list of accepted samples per iteration
    Thetas = [] # list of all samples drawn from all posterios for each iteration [[[theta11,theta12,...,theta1n],[theta21,theta22,...,theta2n]],[[theta11,theta12,...],[]]]
    verbosity = 0 # level of debug messages

    def set_simulator(self, sim):
        """func doc"""
        if callable(sim) or sim is None:
            self.simulator = sim
        else:
            self._eprint("{}: Passed argument {} is not a callable function!".format(type(self).__name__, sim))
            sys.exit(1)

    def set_priors(self, priors):
        """func doc"""
        priors = np.atleast_1d(priors)
        if all(callable(p) for p in priors) or len(priors) == 0:
            self.priors = priors
        else:
            self._eprint("{}: Passed argument {} is not a callable function!".format(type(self).__name__, priors))
            sys.exit(1)

    def set_summaries(self, summaries):
        """func doc"""
        summaries = np.atleast_1d(summaries)
        if all(callable(s) for s in summaries) or len(summaries) == 0:
            self.summaries = summaries
        else:
            self._eprint("{}: Passed argument {} is not a callable function or list of functions!".format(type(self).__name__, summaries))
            sys.exit(1)

    def add_summary(self, summary):
        """func doc"""
        if callable(summary):
            self.summaries = np.hstack((self.summaries, summary))
        else:
            self._eprint("{}: Passed argument {} is not a callable function!".format(type(self).__name__, summary))
            sys.exit(1)

    def add_prior(self, prior):
        """func doc"""
        if callable(prior):
            self.priors = np.hstack((self.priors, prior))
        else:
            self._eprint("{}: Passed argument {} is not a callable function!".format(type(self).__name__, prior))
            sys.exit(1)

    def set_discrepancy(self, disc):
        """func doc"""
        if callable(disc) or disc is None:
            self.discrepancy = disc
        else:
            self._eprint("{}: Passed argument {} is not a callable function!".format(type(self).__name__, disc))
            sys.exit(1)

    def set_observation(self, obs):
        """func doc"""
        if obs is None or (type(obs) == list and len(obs) == 0):
            self.observation = np.empty(0)
        else:
            try:
                obs = np.atleast_1d(obs)
                self.observation = obs
            except (TypeError, ValueError, RuntimeError):
                self._eprint("{}: Passed argument {} cannot be parsed by numpy.atleast_1d()!".format(type(self).__name__, obs))
                sys.exit(1)

    def set_verbosity(self, lvl):
        """set verbosity level of print messages. Possible values are 0, 1, and 2"""
        if type(lvl) == int and lvl >= 0 and lvl <= 2:
            self.verbosity = lvl
        else:
            self._eprint("{}: Passed argument {} has to be integer and between [0,2].".format(type(self).__name__, lvl))
            sys.exit(1)

    @abc.abstractmethod
    def sample(self):
        pass

    @abc.abstractmethod
    def plot_marginals(self):
        pass

    def _eprint(self, *args, **kwargs):
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

    def __init__(self, priors=[], simulator=None, observation=None, discrepancy=None, summaries=[], verbosity=1):
        """constructor"""
        self.set_priors(priors)
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
            self._eprint("{}: Passed argument {} has to be float or a list of floats.".format(type(self).__name__, thresholds))

        if type(nr_samples) == int:
            self.nr_samples = nr_samples
        else:
            self._eprint("{}: Passed argument {} has to be integer.".format(type(self).__name__, nr_samples))

        if self.simulator is None or len(self.priors) == 0 or len(self.observation) == 0 or (len(self.summaries) == 0 and self.discrepancy is None):
            self._eprint("{}: Method sample() called before all necessary functions are set (prior, simulatior, observation, summaries).".format(type(self).__name__))
            return

        print("Rejection sampler started with thresholds: {} and number of samples: {}".format(self.thresholds, self.nr_samples))
        run = 0
        self.nr_iter = [] # reset
        self.Thetas = [] # reset

        for epsilon in self.thresholds:
            X = self.observation
            nr_iter = 0
            start = time.clock()
            thetas = [[] for i in range(len(self.priors))] # for each theta, one array of all sampled thetas

            for i in range(self.nr_samples):
                while True:
                    nr_iter += 1
                    thetas_prop = [p() for p in self.priors] # draw as many thetas as there are priors
                    Y = self.simulator(*thetas_prop) # unpack thetas as single arguments for simulator

                    if self.discrepancy is None:
                        stat_vec_x = np.hstack((s(X) for s in self.summaries))
                        stat_vec_y = np.hstack((s(Y) for s in self.summaries))
                        d = np.linalg.norm(stat_vec_x - stat_vec_y)
                        if d < epsilon:
                            for val,val_prop in zip(thetas, thetas_prop):
                                val.append(val_prop)

                            break
                    else:
                        d = self.discrepancy(X,Y)
                        if d < epsilon:
                            for val,val_prop in zip(thetas, thetas_prop):
                                val.append(val_prop)

                            break

            end = time.clock()
            run += 1
            self.nr_iter.append(nr_iter)
            self.Thetas.append(thetas)

            if self.verbosity == 1:
                print("Run: %2d - Samples: %6d - Threshold: %.2f - Iterations: %10d - Time: %8.2f s" % (run, self.nr_samples, epsilon, self.nr_iter[-1], end - start))

    def plot_marginals(self, names=[]):
        """func doc"""
        if len(self.Thetas) == 0:
            self._eprint("{}: Method plot_marginals() called before sampling was done".format(type(self).__name__))

        for epsilon in self.thresholds:

            nr_plots = len(self.Thetas[-1])
            fig, ax = plt.subplots(1, nr_plots)

            for plot_id, hist in enumerate(self.Thetas[-1]):
                _ax = ax[plot_id]
                _ax.hist(hist, edgecolor="k", bins='auto')
                if names:
                   _ax.set_xlabel(names[plot_id])

            fig.suptitle("Posterior for all model parameters that with\n" + r"$\rho(S(X),S(Y)) < {}, n = {}$".format(epsilon, self.nr_samples))
            plt.show()

    def __str__(self):
        return "{} - priors: {} - simulator: {} - summaries: {} - observation: {} - discrepancy: {} - verbosity: {}".format(
            type(self).__name__, len(self.priors), self.simulator, len(self.summaries), self.observation.shape, self.discrepancy, self.verbosity
        )

"""class doc"""
class SMCSampler(object):
    pass


if __name__ == '__main__':
    print('Subclass:', issubclass(RejectionSampler,
                                  BaseSampler))
    print('Instance:', isinstance(RejectionSampler(),
                                  BaseSampler))
