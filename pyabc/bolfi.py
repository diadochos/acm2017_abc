from .sampler import BaseSampler
from .utils import flatten_function
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from GPyOpt.methods import BayesianOptimization
import numpy as np

class BOLFI(BaseSampler):

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, domain):
        if isinstance(domain, list):
            if len(domain) == len(self.priors):
                self._domain = domain
            else:
                raise ValueError('"domain" needs to contain a tuple for every prior in "priors"!')
        else:
            raise TypeError('"domain" needs to be a list!')

    def __init__(self, priors, simulator, observation, summaries, domain, distance='euclidean', verbosity=1, seed=None):

        # call BaseSampler __init__
        super().__init__(priors, simulator, observation, summaries, distance, verbosity, seed)

        self.domain = domain

    def sample(self):

        # summary statistics of the observed data
        stats_x = flatten_function(self.summaries, self.observation)

        # define distance function
        f = lambda thetas: self.distance(stats_x, flatten_function(self.summaries, self.simulator(*thetas)))

        # create initial evidence set
        # evidence_theta = self.priors.sample(10)
        # evidence_f = np.apply_along_axis(f, axis=1, arr=evidence_theta)

        bounds = [{'name': p.name, 'type': 'continuous', 'domain': domain} for p, domain in zip(self.priors, self.domain)]


        optim = BayesianOptimization(f=f, domain=bounds, acquisition_type='EI',
                                     exact_feval=True, model_type='GP',
                                     num_cores=-1, initial_design_numdata=10,
                                     initial_design_type='sobol')

        max_iter = 30    # evaluation budget
        max_time = 60     # time budget
        eps      = 10e-6  # Minimum allows distance between the las two observations

        optim.run_optimization(max_iter, max_time, eps)

        return optim.x_opt



    def _reset(self):
        """reset class properties for a new call of sample method"""
        self._nr_iter = 0
        self._Thetas = np.empty(0)
