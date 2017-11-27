from .utils import scipy_from_str, numpy_sampler_from_str
from functools import partial
import scipy.stats as ss
import numpy as np


class Prior():
    """Abstract base class for all prior distributions.
    Basically a wrapper for scipy distributions, which uses the corresponding
    numpy sampling function for performance reasons.

    Allows sampling from the prior and evaluating the pdf.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    @property
    def name(self):
        base_name = self._name
        names = []
        if self.multivariate():
            for i in range(len(self)):
                names.append("{}_{}".format(base_name, i + 1))

            return names
        else:
            return self._name


    @name.setter
    def name(self, name):
        if name is None:
            name = ''
        elif not isinstance(name, str):
            raise TypeError("Passed argument {} has to be str.".format(name))

        self._name = name


    def __init__(self, scipy_dist, *args, name=None):
        """Initialize the scipy and numpy objects.

        Args:
            name (str): name of model parameter
            scipy_dist (str): name of the distribution. Can be one of the distributions
                from the scipy.stats module
            *args: the distribution's parameters. See scipy documentation
        """
        self.name = name

        # maybe implement custom functions in the future
        if not isinstance(scipy_dist, str):
            raise TypeError("Passed argument {} has to be str.".format(scipy_dist))

        try:
            # set the distribution to the corresponding scipy object
            self.distribution = scipy_from_str(scipy_dist)(*args)

            try:
                # try to set the sampler to the numpy function if it exists
                # because numpy samplers are faster than scipy
                sampler = numpy_sampler_from_str(scipy_dist, *args)

            # if that fails, fall back to the scipy function
            except:
                sampler = self.distribution.rvs

            # if the prior is multivariate, the samples need to be returned
            # in a transposed format for the rejection sampler to work
            if self.multivariate():
                self._sample = lambda s: sampler(size=s).T
            else:
                self._sample = sampler

        except TypeError:
            # if arguments do not fit the scipy distribution
            raise ValueError('The provided arguments have to be valid for the specified scipy distribution.')
        except AttributeError:
            # if the scipy distribution does not exist
            raise ValueError('"{}" is not a valid scipy distribution.'.format(scipy_dist))


    def multivariate(self):
        return isinstance(self.distribution, ss._multivariate.multi_rv_frozen)


    def __len__(self):
        if self.multivariate():
            return self.sample().shape[0]
        else:
            return 1


    def sample(self, size=None):
        return self._sample(size)


    def pdf(self, theta):
        return self.distribution.pdf(theta)


    def logpdf(self, theta):
        return self.distribution.logpdf(theta)

class PriorList(list):

    def sample(self, size):
        return np.vstack([p.sample(size) for p in self]).T

    def pdf(self, theta):
        return np.prod([p.pdf(theta) for p in self])

    def logpdf(self, theta):
        return sum([p.logpdf(theta) for p in self])

    def __len__(self):
        return sum([len(p) for p in self])
