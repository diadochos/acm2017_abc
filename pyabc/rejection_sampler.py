from .sampler import BaseSampler
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
        self._acceptance_rate = 0
        self._Thetas = np.empty(0)

    def sample_from_priors(self):
        """draw samples from all priors and return as list of outputs

        :return list of outputs for each prior
        """
        return [p() for p in self.priors]

    def _flatten_function(self, list_of_f, args=None):
        """return function output as 1d array"""
        ret = np.empty(0)
        for f in list_of_f:
            if args is None:
                ret = np.concatenate((ret, np.atleast_1d(f()).flatten()))
            else:
                ret = np.concatenate((ret, np.atleast_1d(f(args)).flatten()))

        return ret

    def _flatten_output(self, x):
        return np.hstack(np.atleast_1d(e).flatten() for e in x)

    def _run_rejection_sampling(self, nr_samples):
        """the abc rejection sampling algorithm with batches"""

        X = self.observation

        list_of_stats_x = self._flatten_function(self.summaries, X)

        thetas = np.zeros((nr_samples, sum(np.atleast_1d(a).shape[0] for a in self.sample_from_priors())))

        nr_iter = 0
        start = time.clock()
        for i in range(nr_samples):
            while True:
                nr_iter += 1

                thetas_prop = self.sample_from_priors()  # draw as many thetas as there are priors
                Y = self.simulator(*thetas_prop)  # unpack thetas as single arguments for simulator
                list_of_stats_y = self._flatten_function(self.summaries, Y)

                if any(s1.shape != s2.shape for s1,s2 in zip(list_of_stats_x, list_of_stats_y)):
                    raise ValueError("Dimensions of summary statistics for observation X ({}) and simulation data Y ({}) are not the same".format(list_of_stats_x, list_of_stats_y))

                # either use predefined distance function or user defined discrepancy function
                d = self.distance(list_of_stats_x, list_of_stats_y)

                if d < self.threshold:
                    thetas[i, :] = self._flatten_output(thetas_prop)
                    break


        self._runtime = time.clock() - start

        self._acceptance_rate = nr_samples / nr_iter
        self._Thetas = thetas
        return self.Thetas

    def _run_rejection_sampling_opt(self, nr_samples, batch_size=100):
        """the abc rejection sampling algorithm with batches"""

        X = self.observation

        list_of_stats_x = self._flatten_function(self.summaries, X)

        accepted_thetas = []

        start = time.clock()

        nr_batches = 0

        while len(accepted_thetas) < nr_samples:
            nr_batches += 1
            # draw batch_size parameters from priors
            thetas_batch = [self.sample_from_priors() for i in range(batch_size)]
            Y_batch = (self.simulator(*thetas) for thetas in thetas_batch)
            summaries_batch = (self._flatten_function(self.summaries, Y) for Y in Y_batch)
            d_batch = (self.distance(list_of_stats_x, list_of_stats_y) for list_of_stats_y in summaries_batch)
            accepted_thetas.extend((theta for theta, d in zip(thetas_batch, d_batch) if d < self.threshold))


        accepted_thetas = accepted_thetas[:nr_samples]
        thetas = np.array(accepted_thetas)

        self._runtime = time.clock() - start

        self._acceptance_rate = nr_samples / (nr_batches * batch_size)
        self._Thetas = thetas


    def sample(self, threshold, nr_samples, batch_size=100, old_version=False):
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
        if old_version:
            self._run_rejection_sampling(nr_samples)
        else:
            self._run_rejection_sampling_opt(nr_samples, batch_size)

        if self.verbosity == 1:
            print("Samples: %6d - Threshold: %.2f - Acceptance rate: %10d %% - Time: %8.2f s" % (nr_samples, self.threshold, self.acceptance_rate*100, self.runtime))

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
