import os
import sys
sys.path.append(os.path.abspath(os.path.join('../..')))
import pyabc
import numpy as np
import matplotlib.pyplot as plt
from pyabc.examples import twenty_gauss
from pyabc.plots import plot_marginals, plot_pairs
from pyabc.prior import PriorList

simulator = twenty_gauss.simulator
#distance = multiplebivariategauss.MultipleBivariateGauss._distance
summary = twenty_gauss.summaries


np.random.seed(1337)

# data parameters
def generateMus(size):
    mus = [None] * size 
    for i in range(size):
        mus[i] = [np.random.randint(0,10), np.random.randint(0,10)]
    return mus
    

mus = generateMus(10)
N = 50

mus = np.array(mus)
x = simulator(*mus.flatten())

# prior parameters
prior_mu = pyabc.Prior('uniform', 0, 10, name='mu')
prior_mus = [prior_mu] * 20
#prior_mus = [pyabc.Prior('normal', m, 1.0, name='mu') for m in mus.flatten()]

mus = np.array(mus)
x = simulator(*mus.flatten())

def distance(s1,s2):
    diff = s1 - s2 
    rmsd = np.mean(diff*diff)
    return np.sqrt(rmsd) 

rej = pyabc.RejectionSampler(
    priors=prior_mus,
    simulator=simulator,
    summaries=summary,
    distance=distance,
    observation=x,
    verbosity=2
)

smc = pyabc.SMCSampler(
    priors=prior_mus,
    simulator=simulator,
    summaries=summary,
    distance=distance,
    observation=x
)

mcmc = pyabc.MCMCSampler(
    priors=prior_mus,
    simulator=simulator,
    summaries=summary,
    distance=distance,
    observation=x
)

# abcde = pyabc.ABCDESampler(
#     priors=prior_mus,
#     simulator=simulator,
#     summaries=summary,
#     distance=distance,
#     observation=x
# )


bolfi = pyabc.BOLFI(
    priors=prior_mus,
    simulator=simulator,
    summaries=summary,
    distance=distance,
    observation=x,
    domain = [(0,10)] * 20
)

path_to_data = "D:\\Dropbox\\Dropbox\\AppliedCognitiveModelling2017\\ABC\\03 data"

nr_samples, threshold, batch_size = 10000, 1, 10000
rej.sample(threshold=threshold, nr_samples=nr_samples, batch_size = batch_size)
fname = os.path.join(path_to_data, "ex05", "rej_{}_{}_{}.pkl".format(nr_samples, str(threshold).replace(".", "_"), batch_size))
rej.save(fname)

