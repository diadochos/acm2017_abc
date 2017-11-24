import pyabc
import numpy as np

# multivaraite test
mu = np.array([1,2])
sigma = np.diag([1,1])

y0 = np.random.multivariate_normal(mu, sigma, 10)

# prior_mu = pyabc.Prior('multivariate_normal', np.array([0,0]), np.diag([1,1]))

p1 = pyabc.Prior('normal', 0, 1)
p2 = pyabc.Prior('normal', 0, 1)

def simulator(mu1, mu2):
    return np.random.multivariate_normal(np.array([mu1, mu2]), sigma, 10)

def mean(x):
    return np.mean(x, 0)

def var(x):
    return np.cov(x.T)

rej_samp = pyabc.RejectionSampler(priors=[p1, p2], simulator=simulator, summaries=[mean, var], observation=y0)

rej_samp.sample(0.5, 100, batch_size=1000)
