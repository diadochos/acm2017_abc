# sampler.py

import abc

import numpy as np

from .prior import Prior, PriorList


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
            lambda x, y: np.linalg.norm(x - y)
    }

    def __init__(self, priors, simulator, observation, summaries, distance, verbosity, seed):

        self.priors = priors
        self.simulator = simulator
        self.observation = observation
        self.summaries = summaries

        # optional
        self.verbosity = verbosity
        self.distance = distance

        if seed is not None:
            np.random.seed(seed)

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
        if all(issubclass(type(p), Prior) for p in priors):
            self._priors = PriorList(priors)
        else:
            print(all(issubclass(p, Prior) for p in priors))
            raise TypeError("Passed argument {} is not a subclass of prior!".format(priors))

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
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, d):
        """func doc"""
        if callable(d):
            self._distance = d
        elif isinstance(d, str):
            if d in self.__distances.keys():
                self._distance = self.__distances[d]
            else:
                raise KeyError("Passed argument {} is not a valid distance function. Choose from {}.".format(d,
                                                                                                             self.__distances.keys()))
        else:
            raise TypeError(
                "Passed argument {} is neither a callable function nor a name of predefined distance functions!".format(
                    d))

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
        if isinstance(nr_samples, (int, float)):
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
        return self.__distance

    @property
    def acceptance_rate(self):
        return self._acceptance_rate

    @property
    def distances(self):
        return self._distances

    @abc.abstractmethod
    def sample(self):
        pass

    @abc.abstractmethod
    def _reset(self):
        pass
