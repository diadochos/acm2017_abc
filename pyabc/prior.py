from .utils import scipy_from_str

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
        if isinstance(name, str):
            try:
                self.distribution = scipy_from_str(name)(*args)
            except TypeError:
                raise ValueError('The provided arguments have to be valid for the specified scipy distribution.')
            except AttributeError:
                raise ValueError('"{}" is not a valid scipy distribution.'.format(name))
        else:
            raise TypeError("Passed argument {} has to be str.".format(name))


    def sample(self, size=None):
        return self.distribution.rvs(size)

    def pdf(self, theta):
        return self.distribution.pdf(theta)

    # @abc.abstractmethod
    # def plot_marginal(self):
    #     pass
