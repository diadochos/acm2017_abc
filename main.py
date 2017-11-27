import numpy as np
import pylab as plt
from collections import Counter
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
from pyabc.examples import tuberculosis
import pyabc.prior
from pyabc.plots import plot_marginals, plot_particles


alpha = 0.2
delta = 0
tau = 0.198
m = 20

prior_alpha = pyabc.Prior("uniform", 0.1, 1, name="alpha")
prior_tau = pyabc.Prior("uniform", 0.01, 1, name="tau")
domain = [(0.1, 1), (0.01, 1)]

simulator = tuberculosis.simulator

for i in range(100):
    params = [prior_alpha.sample(), prior_tau.sample()]
    y0 = simulator(*params)

bolfi = pyabc.BOLFI(
    priors=[prior_alpha, prior_tau],
    simulator=simulator,
    summaries=tuberculosis.summaries,
    observation=y0,
    seed = 1337,
    domain=domain
)

bolfi.sample(nr_samples=10000, n_chains=10)