from pyabc import BOLFI, Prior
from pyabc.prior import PriorList
import numpy as np

mu0 = 3.5
y0 = np.random.normal(mu0, 1, 10)

prior = Prior('uniform', 0, 5)

def simulator(mu):
    return np.random.normal(mu, 1, 10)

bolfi = BOLFI(priors=[prior], simulator=simulator, observation=y0, summaries=[np.mean], domain=[(-1,1)])
bolfi.sample()
