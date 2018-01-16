import time

import numpy as np

from .rejection_sampler import RejectionSampler
from .sampler import BaseSampler
from .utils import flatten_function


class MCMCSampler(BaseSampler):
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

        # call BaseSampler __init__
        super().__init__(priors, simulator, observation, summaries, distance, verbosity, seed)

    def _reset(self):
        """reset class properties for a new call of sample method"""
        self._nr_iter = 0
        self._Thetas = np.empty(0)
        self._simtime = 0

    def _run_mcmc_sampling(self, step_size):
        X = self.observation
        stats_x = flatten_function(self.summaries, X)
        num_priors = len(self.priors)

        thetas = np.zeros((self.nr_samples, num_priors))
        distances = np.zeros(self.nr_samples)

        nr_iter = 0

        start = time.clock()

        rej_samp = RejectionSampler(
            priors=self.priors.tolist(),
            simulator=self.simulator,
            summaries=self.summaries,
            distance=self.distance,
            observation=self.observation,
            verbosity=0
        )

        # get the best from 10 samples to initialize the chain
        rej_samp.sample(threshold=self.threshold, nr_samples=10, batch_size=10)

        thetas[0] = rej_samp.Thetas[np.argmin(rej_samp.distances)]
        distances[0] = np.min(rej_samp.distances)

        if step_size is None:
             step_size = 2 * np.cov(rej_samp.Thetas.T)

        step = np.identity(num_priors) * step_size

        for i in range(1, self.nr_samples):
            while True:
                nr_iter += 1
                theta = thetas[i - 1, :]
                thetap = np.random.multivariate_normal(theta, np.atleast_2d(step))

                # for which theta pertubation produced unreasonable values?
                for id, prior in enumerate(self.priors):
                    if prior.pdf(thetap[id]) == 0:
                        thetap[id] = theta[id]

                Y = self.simulate((np.atleast_1d(thetap)))  # unpack thetas as single arguments for simulator
                stats_y = flatten_function(self.summaries, Y)

                # either use predefined distance function or user defined discrepancy function
                d = self.distance(stats_x, stats_y)

                if d <= self.threshold:
                    A = self.priors.pdf(thetap) / self.priors.pdf(theta)
                    u = np.random.uniform(0, 1)

                    if u < A:
                        thetas[i, :] = thetap
                        distances[i] = d
                    else:
                        thetas[i, :] = theta
                        distances[i] = distances[i - 1]
                    break
                    # step_size = np.cov(thetas[0:i+1,:].T)
                
                self._nr_iter = nr_iter
                self._runtime = time.clock() - start
                self._acceptance_rate = self.nr_samples / self.nr_iter
            
                if nr_iter % 1000 == 0:
                    self.log(thetas[thetas != 0], False)

        self._runtime = time.clock() - start
        self._nr_iter = nr_iter
        self._acceptance_rate = self.nr_samples / self.nr_iter
        self._Thetas = thetas
        self._distances = distances

        return thetas

    def sample(self, threshold, nr_samples, step_size=None):
        """Main method of sampler. Draw from prior and simulate data until nr_samples were accepted according to threshold.

        Args:
            threshold: Threshold is used as acceptance criteria for samples.
            nr_samples: Number of samples drawn from prior distribution.
            step_size: step size between thetas within MCMC

        Returns:
            Nothing

        """
        self.threshold = threshold
        self.nr_samples = nr_samples

        if step_size is not None and len(step_size) != len(self.priors):
            raise ValueError('Step size for every prior is required')

        if self.verbosity:
            print(
                "MCMC sampler started with threshold: {} and number of samples: {}".format(self.threshold, self.nr_samples))

        self._reset()

        # RUN ABC REJECTION SAMPLING
        self._run_mcmc_sampling(step_size)

        self.log(final=True)

    def log(self, accepted_thetas=[], final=False):

        if self.verbosity > 1 and not final:
            print("Samples: %6d / %6d (%3d %%)- Threshold: %.4f - Iterations: %10d - Acceptance rate: %4f - Time: %8.2f s" % (
                len(accepted_thetas),
                self.nr_samples,
                int(np.round(len(accepted_thetas) / self.nr_samples * 100)), 
                self.threshold, 
                self.nr_iter, 
                self.acceptance_rate, 
                self.runtime)
            )

        if final:
            print("Samples: %6d - Threshold: %.4f - Iterations: %10d - Acceptance rate: %4f - Time: %8.2f s" % (
                self.nr_samples, 
                self.threshold, 
                self.nr_iter, 
                self.acceptance_rate,
                self.runtime)
            )

    def __str__(self):
        return "{} - priors: {} - simulator: {} - summaries: {} - observation: {} - discrepancy: {} - verbosity: {}".format(
            type(self).__name__, len(self.priors), self.simulator, len(self.summaries), self.observation.shape,
            self.discrepancy, self.verbosity
        )
