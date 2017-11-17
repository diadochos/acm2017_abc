import pyabc
import numpy as np

# multivaraite test
mu = np.array([1,2])
sigma = np.diag([1,1])

y0 = np.random.multivariate_normal(mu, sigma, 10)

def prior_mu():
    return np.random.uniform(0,2,2)

def simulator(mu):
    return np.random.multivariate_normal(mu, sigma, 10)

def mean(x):
    return np.mean(x, 0)

def var(x):
    return np.cov(x.T)

rej_samp = pyabc.RejectionSampler(priors=prior_mu, simulator=simulator, summaries=[mean, var], observation=y0)

rej_samp.sample(0.5, 100)
