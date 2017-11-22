from .utils import scipy_from_str, numpy_sampler_from_str
from functools import partial
import scipy.stats as ss


class Prior():
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

    def __init__(self, name, *args):
        if not isinstance(name, str):
            raise TypeError("Passed argument {} has to be str.".format(name))

        try:
            # set the distribution to the corresponding scipy object
            self.distribution = scipy_from_str(name)(*args)

            try:
                # try to set the sampler to the numpy function if it exists
                # because numpy samplers are faster than scipy
                sampler = numpy_sampler_from_str(name, *args)

            # if that fails, fall back to the scipy function
            except:
                sampler = self.distribution.rvs

            if isinstance(self.distribution, ss._multivariate.multi_rv_frozen):
                self._sample = lambda s: sampler(size=s).T
            else:
                self._sample = sampler

        except TypeError:
            # if arguments do not fit the scipy distribution
            raise ValueError('The provided arguments have to be valid for the specified scipy distribution.')
        except AttributeError:
            # if the scipy distribution does not exist
            raise ValueError('"{}" is not a valid scipy distribution.'.format(name))





    def sample(self, size=None):
        return self._sample(size)

    def pdf(self, theta):
        return self.distribution.pdf(theta)

    # @abc.abstractmethod
    # def plot_marginal(self):
    #     pass
