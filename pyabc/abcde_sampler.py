import time

import numpy as np
import scipy.stats as ss
import pyabc

from .sampler import BaseSampler
from .utils import flatten_function

"""class doc"""


class ABCDESampler(BaseSampler):

    @property
    def weights( self ):
        return self._weights


    @property
    def threshold( self ):
        return self._threshold


    def __init__( self, priors, simulator, observation, summaries, exp_lambda = 20, distance='euclidean', verbosity=1, seed=None ):
        # call BaseSampler __init__
        # extend list of priors by prior for delta
        exponential_prior = pyabc.Prior('expon', 0, 1/exp_lambda, name="delta")
        if not isinstance(priors, list):
            priors = [priors]
        priors.append(exponential_prior)  # now drawing samples from priors means to draw sample for delta, too
        super().__init__(priors, simulator, observation, summaries, distance, verbosity, seed)
        self._nr_priors = len(self.priors)


    def sample( self, nr_iter, nr_samples, nr_groups, burn_in, alpha=0.1, beta=0.1, kappa=0.9):
        """Draw samples using the genetic ABCDE Algorithm

        Args:
            nr_iter: Number of iterations of the algorithm
            nr_groups: Number of population pools
            nr_samples: Number of samples we want to obtain from theta posterior
  			burn_in: Number of iterations in 'burn_in' phase
            alpha: Probability to do the migration step
            beta: Probability to do the mutation step
            kappa: Probability to keep the newly generated theta component
            exp_lambda: Value that is used to draw delta from exponential distribution


        Returns:
            Nothing

        """
        if (nr_samples % nr_groups) != 0:
            raise ValueError("Nr samples must be divided evenly into", nr_groups, 'groups')

        if (burn_in > nr_iter):
            raise ValueError("Burn in must be smaller then the total number of iterations")

        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa

        self._nr_iter = nr_iter
        self._burn_in = burn_in
        self._nr_groups = nr_groups
        self._pool_size = int(nr_samples / nr_groups)  # number of particles per group

        print("ABC-Differential-Evolution sampler started with number of samples: {}".format(nr_samples))

        self._reset()
        self._run_ABCDE_sampling()

        if self.verbosity == 1:
            print("Samples: %6d - Iterations: %10d - Acceptance rate: %4f - Time: %8.2f s" % (
                nr_samples, self.nr_iter, self.acceptance_rate, self.runtime))


    def calculate_fitness( self, curr_theta, distance ):
        """Eq (3) extended by delta prior from paper, calculate fitness of proposed theta"""

        error_distribution = ss.norm(0, curr_theta[
            -1])  # delta is always last entry of theta vector and after burn in equals group delta
        fitness = self.priors.pdf(curr_theta) * error_distribution.pdf(distance)

        return fitness


    def mh_step( self, it, group, i, theta_star ):
        """perform the metropolis hasting update step, Eq (6) in the paper """
        theta_old = self._particles[it - 1, group, i]

        Y = self.simulator(*(np.atleast_1d(theta_star[:-1])))  # unpack thetas as single arguments for simulator
        stats_y = flatten_function(self.summaries, Y)
        d = self.distance(self._stats_x, stats_y)

        # only change the delta parameter during burn_in and take the groups delta during sample mode
        if self._sampling_mode == 'sample':
            theta_star[-1] = self._group_deltas[group]

        proposal_fitness = self.calculate_fitness(theta_star, d)
        previous_fitness = self._weights[it - 1, group, i]

        # calculate MH probability
        MH_prob = min(1, proposal_fitness / previous_fitness)
        u = np.random.uniform(0, 1)

        if u < MH_prob:
            self._particles[it, group, i, :] = theta_star
            self._distances[it, group, i] = d
            self._weights[it, group, i] = proposal_fitness
        else:
            self._particles[it, group, i, :] = theta_old
            self._distances[it, group, i] = self._distances[it - 1, group, i]
            self._weights[it, group, i] = previous_fitness


    def crossover( self, it, group ):
        """This generates new particles, by operating in the vector space of thetas for a particular particle cluster"""
        for i in range(self._pool_size):
            while True:
                # choose params we use for linear combination
                y1 = np.random.uniform(0.5, 1)
                y2 = 0
                if self._sampling_mode == 'burn_in':
                    y2 = np.random.uniform(0.5, 1)
                b = np.random.uniform(-0.001, 0.001)  # TODO make this a model parameter(?)

                # index party!
                # get indices for all particles
                idx_b = np.random.choice(np.arange(self._pool_size), p=self._weights[it - 1, group, :])
                idx_all = np.arange(self._pool_size)
                idx_all = np.delete(idx_all, [i, idx_b])
                idx_m, idx_n = np.random.choice(idx_all, 2)

                # get the 3 theta vectors used for crossover
                theta_t = self._particles[it - 1, group, i]
                theta_b = self._particles[it - 1, group, idx_b]  # base_particle
                theta_m = self._particles[it - 1, group, idx_m]
                theta_n = self._particles[it - 1, group, idx_n]

                # find a new theta, as a linear combination in the vector space of thetas within the cluster
                theta_star = theta_t + y1 * (theta_m - theta_n) + y2 * (theta_b - theta_t) + b

                # keep some of the old features with probability (1-k)
                reset_probabilities = np.random.uniform(0, 1, size=len(theta_star))
                for j in range(len(theta_star)):
                    if reset_probabilities[j] < (1 - self._kappa):
                        theta_star[j] = theta_t[j]

                # make sure we found a great theta that works with our prior and then we can simulate and see how well it fits the data
                if self.priors.pdf(theta_star) > 0:
                    self.mh_step(it, group, i, theta_star)

                break


    def mutate( self, it, group ):
        """ This slightly perturbs each particle within a group using a perturbation kernel (mv-gaussian)"""
        # TODO: is this really the best method to choose the step_size(?)
        self._sigmas[it, group, :] = 2 * np.cov(self._particles[it - 1, group].T, aweights=self._weights[it - 1, group, :])
        sigma = self._sigmas[it, group, :]

        for i in range(self._pool_size):
            while True:
                theta_old = self._particles[it - 1, group, i, :]
                theta_star = np.atleast_1d(ss.multivariate_normal(theta_old, sigma, allow_singular=True).rvs())

                # make sure we found a great theta that works with our prior and then we can simulate and see how well it fits the data
                if self.priors.pdf(theta_star) > 0:
                    self.mh_step(it, group, i, theta_star)

                break


    def migrate( self, it ):
        """Migrate between the groups such as to diversify them, this happens by choosing K groups and cycling the weakest particle to the neighbouring group"""
        K = np.random.randint(1, self._nr_groups + 1)  # how many groups
        groups = np.arange(self._nr_groups)
        np.random.shuffle(groups)
        groups = groups[:K]  # which groups

        # setup arrays to temporarily store the indices of the particles we want to swap
        weak_particles_idx = []

        # first choose all the weak_particles by their inverse of their weights and store their index
        for g in groups:
            # normalize the inverse group weights
            group_weights = 1 / (self.weights[it - 1, g, :] + 1e-12)  # dont divide by zero so add bias(?)
            group_weights = group_weights / np.sum(group_weights)
            weak_particles_idx.append(np.random.choice(np.arange(self._pool_size), p=group_weights))

        # store all weakest thetas
        list_of_weak_thetas_and_weights = []
        for g, idx in zip(groups, weak_particles_idx):
            list_of_weak_thetas_and_weights.append(
                (self._particles[it - 1, g, idx], self._weights[it - 1, g, idx])
            )

        # overwrite each weakest theta and weight with the weakest theta and weight from the next grp
        for idx, g, weak_idx in zip(range(K), groups, weak_particles_idx):
            # temporarily store our particles before we overwrite them with the previous
            self._particles[it - 1, g, weak_idx] = list_of_weak_thetas_and_weights[idx - 1][0]
            self._weights[it - 1, g, weak_idx] = list_of_weak_thetas_and_weights[idx - 1][1]


    def init_thetas( self ):
        """initialize theta from prior for each group"""

        for i in range(self._nr_groups):
            self._particles[0, i, :, :] = self.priors.sample(self._pool_size)
            for j in range(self._pool_size):
                curr_theta = self._particles[0, i, j, :]

                Y = self.simulator(*(np.atleast_1d(curr_theta[:-1])))  # unpack thetas as single arguments for simulator
                stats_y = flatten_function(self.summaries, Y)
                d = self.distance(self._stats_x, stats_y)

                self._weights[0, i, j] = self.calculate_fitness(curr_theta, d)  # TODO: different deltas necessary?
                self._distances[0, i, j] = d
                # normalize weights within pool
            self._weights[0, i, :] = self._weights[0, i, :] / sum(self._weights[0, i, :])


    def _run_ABCDE_sampling( self):
        X = self.observation
        self._stats_x = flatten_function(self.summaries, X)

        self._sampling_mode = 'burn_in'

        start = time.clock()

        for t in range(self.nr_iter):
            # init particle pools by simply sampling from the prior for each pool
            if t == 0:
                if self.verbosity:
                    print('initializing pools')
                self.init_thetas()

            else:
                if self.verbosity:
                    print('starting iteration[ %4d ]' % (t))

                if t == self._burn_in:
                    self._sampling_mode = 'sample'
                    # after burn in find min delta of each group
                    for g in range(self._nr_groups):
                        self._group_deltas[g] = self._particles[:t, g, :, -1].min()

                p1 = np.random.uniform(0, 1)

                if p1 < self._alpha:
                    # Migrate between the groups such as to diversify them, this happens by choosing n groups and cycling the weakest particle to the neighbouring group
                    self.migrate(t)

                    # for each pool of particles do the following:
                for i in range(self._nr_groups):
                    p2 = np.random.uniform(0, 1)
                    if p2 < self._beta:
                        # This slightly perturbs each particle within a group using a perturbation kernel (mv-gaussian)
                        self.mutate(t, i)
                    else:
                        # This generates new particles, by operating in the vector space of thetas for a particular particle cluster
                        self.crossover(t, i)

        self._runtime = time.clock() - start
        self._Thetas = self._thetas[-1, :, :]

        return self._Thetas


    def _reset( self):
        """reset class properties for a new call of sample method"""

        # TODO: only previous and current theta
        self._particles = np.zeros((self.nr_iter, self._nr_groups, self._pool_size,
                                    self._nr_priors))  # number of model parameters plus delta for psi distribution
        self._weights = np.zeros((self.nr_iter, self._nr_groups, self._pool_size))
        self._sigmas = np.zeros((self.nr_iter, self._nr_groups, self._nr_priors, self._nr_priors))
        self._distances = np.zeros((self.nr_iter, self._nr_groups, self._pool_size))
        self._group_deltas = np.zeros(self._nr_groups)
