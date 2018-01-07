import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

from .base_example import Example

# autocovariance
def acov(x, k):
    n = np.size(x)
    x_mean = np.mean(x)
    acov = 0
    for i in np.arange(0, n-k):
        acov += ((x[i+k]) - x_mean) * (x[i] - x_mean)
    return (1/(n-1)) * acov

# autoregression
def autoregression(x):
    y_3 = x**3 - np.mean(x**3)
    y_6 = y_3**2

    y = y_3[1:]
    X = np.vstack([y_3[:-1], y_6[:-1]]).T

    beta_hat = np.linalg.lstsq(X,y)[0]
    return beta_hat

def ofd_regression(x):
    # orderered first differences of observed data
    ofd = np.sort(np.diff(x))
    X = np.vstack([np.ones_like(ofd), ofd, ofd**2, ofd**3]).T
    beta_hat = np.linalg.lstsq(X, ofd)[0]
    return beta_hat

def num_zeros(x):
    return np.count_nonzero(x==0)


class Ricker(Example):

    def simulator(self, log_r, sigma=0.3, phi=10, n=50):
        N = [1]
        for t in range(1, n + 1):
            N_t = N[t-1] * np.exp(log_r + -N[t-1] + sigma * np.random.normal())
            N.append(N_t)

        N = np.array(N)
        y = np.random.poisson(phi * N)
        return y

    def _summaries(self):
        return [lambda x: acov(x, i) for i in range(6)] + [autoregression, ofd_regression, np.mean, num_zeros]


ricker = Ricker()
