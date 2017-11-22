from .sampler import BaseSampler
from .utils import flatten_function
import matplotlib.pyplot as plt
import numpy as np
import time


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

    def __init__(self, priors, simulator, observation, summaries, distance='euclidean', verbosity=1, seed=None):
        """constructor"""
        # must have
        self.priors = priors
        self.simulator = simulator
        self.observation = observation
        self.summaries = summaries

        # optional
        self.verbosity = verbosity
        self.distance = distance

        if seed is not None:
            np.random.seed(seed)

    def _reset(self):
        """reset class properties for a new call of sample method"""
        self._nr_iter = 0
        self._Thetas = np.empty(0)

    def sample_from_priors(self, size):
        """draw samples from all priors and return as list of outputs

        :return list of outputs for each prior
        """
        return np.vstack([p.sample(size) for p in self.priors]).T



    def _run_rejection_sampling(self, nr_samples, batch_size):
        """the abc rejection sampling algorithm with batches"""

        X = self.observation

        stats_x = flatten_function(self.summaries, X)

        summaries = lambda thetas: flatten_function(self.summaries, self.simulator(*thetas))

        accepted_thetas = []

        start = time.clock()

        nr_batches = 0

        while len(accepted_thetas) < nr_samples:
            nr_batches += 1
            # draw batch_size parameters from priors
            thetas_batch = self.sample_from_priors(batch_size)
            summaries_batch = np.apply_along_axis(summaries, axis=1, arr=thetas_batch)
            d_batch = np.apply_along_axis(lambda stats_y: self.distance(stats_x, stats_y), axis=1, arr=summaries_batch)

            accepted_thetas.extend(thetas_batch[d_batch < self.threshold])

        accepted_thetas = accepted_thetas[:nr_samples]
        thetas = np.array(accepted_thetas)

        self._runtime = time.clock() - start

        self._nr_iter = (nr_batches * batch_size)
        self._Thetas = thetas
        return thetas


    def sample(self, threshold, nr_samples, batch_size=1000):
        """Main method of sampler. Draw from prior and simulate data until nr_samples were accepted according to threshold.

        Args:
            threshold: Threshold is used as acceptance criteria for samples.
            nr_samples: Number of samples drawn from prior distribution.
            distance: distance measure to compare summary statistics. (default) euclidean

        Returns:
            Nothing

        """
        self.threshold = threshold

        print("Rejection sampler started with threshold: {} and number of samples: {}".format(self.threshold, nr_samples))

        self._reset()

        # RUN ABC REJECTION SAMPLING
        self._run_rejection_sampling(nr_samples, batch_size)

        if self.verbosity == 1:
            print("Samples: %6d - Threshold: %.2f - Number of iterations: %10d - Time: %8.2f s" % (nr_samples, self.threshold, self.nr_iter, self.runtime))

    def plot_marginals(self, names=[]):
        """func doc"""

        if self.Thetas.shape == (0,):
            raise Warning("Method was called before sampling was done")

        nr_plots = self.Thetas.shape[1] # number of columns

        fig, ax = plt.subplots(1, nr_plots)

        for plot_id, hist in enumerate(self.Thetas.T):
            if nr_plots == 1:
                _ax = ax
            else :
                _ax = ax[plot_id]

            _ax.hist(hist, edgecolor="k", bins='auto', normed=True)
            if names and len(names) == nr_plots:
               _ax.set_xlabel(names[plot_id])

        fig.suptitle("Posterior for all model parameters with\n" + r"$\rho(S(X),S(Y)) < {}, n = {}$".format(self.threshold, self.Thetas.shape[0]))
        plt.show()

    def __str__(self):
        return "{} - priors: {} - simulator: {} - summaries: {} - observation: {} - discrepancy: {} - verbosity: {}".format(
            type(self).__name__, len(self.priors), self.simulator, len(self.summaries), self.observation.shape, self.discrepancy, self.verbosity
        )
