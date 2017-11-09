import pyabc
import numpy as np

rej_samp = pyabc.RejectionSampler()

def prior_mu():
    return np.random.uniform(-2,2)

def prior_sigma():
    return np.random.uniform(1,5)

def simulator(mu, sigma):
    return np.random.normal(mu, sigma, 30)

def summary_mean(X):
    return np.mean(X)

def summary_var(X):
    return np.var(X)

#observation
# Set the generating parameters that we will try to infer
mean0 = 1
std0 = 3

# Generate some data (using a fixed seed here)
np.random.seed(20170525)
y0 = simulator(mean0, std0)

print(y0)
print(summary_mean(y0))
print(summary_var(y0))

# use sampler
# var 1 -> setter
rej_samp.set_priors([prior_mu, prior_sigma])
rej_samp.set_simulator(simulator)
rej_samp.set_summaries([summary_mean])
rej_samp.add_summary(summary_var)
rej_samp.set_observation(y0)

print(rej_samp)

threshold = .5
nr_samples = 1000
rej_samp.sample(threshold, nr_samples)

rej_samp.plot_marginals(["mu", "sigma"])

# var 2 -> constructor
rej_samp = pyabc.RejectionSampler(priors=[prior_mu, prior_sigma], simulator=simulator, summaries=[summary_var, summary_mean], verbosity=1)
rej_samp.sample(threshold, nr_samples)
rej_samp = pyabc.RejectionSampler(priors=[prior_mu, prior_sigma], simulator=simulator, summaries=[summary_var, summary_mean], observation=y0, verbosity=1)
rej_samp.sample(threshold, nr_samples)

rej_samp.plot_marginals(["mu", "sigma"])
