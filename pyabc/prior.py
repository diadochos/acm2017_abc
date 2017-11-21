import abc
import numpy as np
import scipy.stats as ss

class Prior(metaclass=abc.ABCMeta):
    """Abstract base class for all priors. Defines common setters and properties

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

    # set and get priors
    @property
    def distribution(self):
        return self._distribution

    @abc.abstractmethod
    def sample(self):
        pass

    @abc.abstractmethod
    def pdf(self,theta):
        pass

    @abc.abstractmethod
    def plot_marginal(self):
        pass


class UniformPrior(Prior):
    def __init__(self,a,b):
        """constructor"""
        # must have
        self._distribution = ss.uniform(a,b)
        self._a = a
        self._b = b

    def sample(self):
        return np.random.uniform(self._a,self._b)

    def pdf(self,theta):
        return self._distribution.pdf(theta)

    def plot_marginal(self):
        pass


class GaussianPrior(Prior):
    def __init__(self,mu,sigma):
        """constructor"""
        # must have
        self._distribution = ss.norm(mu,sigma)
        self._mu = mu
        self._sigma = sigma

    def sample(self):
        return np.random.norm(self._mu,self._sigma)

    def pdf(self,theta):
        return self._distribution.pdf(theta)

    def plot_marginal(self):
        pass
