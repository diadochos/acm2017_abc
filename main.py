import pyabc
import numpy as np

rej_samp = pyabc.RejectionSampler()

# multivaraite test
mu = np.array([1,2])
sigma = np.diag([1,1])

y0 = np.random.multivariate_normal(mu, sigma, 10)

def prior_mu():
    return np.random.uniform(0,2,2)

def simulator(mu1, mu2):
    return np.random.multivariate_normal([mu1,mu2], sigma, 10)

def mean(x):
    return np.mean(x, 0)

def var(x):
    return np.cov(x.T)

rej_samp.set_priors([prior_mu])
rej_samp.set_simulator(simulator)
rej_samp.set_summaries([mean, var])
rej_samp.set_observation(y0)

rej_samp.sample(0.5, 5000)

