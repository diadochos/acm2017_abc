import numpy as np
from .base_example import Example


def mean(x):
    return np.mean(x, 0)

def cov(x):
    return np.cov(x.T)


class MVGauss(Example):

    def _summaries(self):
        return [mean, cov]

    def simulator(self, mu1, mu2):
        sigma = np.array([[2,0.5],[0.5,1]])
        return np.random.multivariate_normal(np.array([mu1, mu2]), sigma, 10)


mv_gauss = MVGauss()
