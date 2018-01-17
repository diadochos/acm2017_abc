import numpy as np
from .base_example import Example


def group_means(x):
    return np.mean(x, axis=0)

def distance(s1,s2):
    diff = s1 - s2
    rmsd = np.mean(diff*diff)
    return np.sqrt(rmsd)


class MultivariateGauss(Example):

    #summary stat
    def _summaries(self):
        return [group_means]

    def simulator(self, *mus):
        mu = np.array(mus)
        sigma = np.eye(mu.shape[0]) * 0.01**2

        res = np.random.multivariate_normal(mu, sigma, 50)

        return res


    def _distance(self):
        return distance

twenty_gauss = MultivariateGauss()
