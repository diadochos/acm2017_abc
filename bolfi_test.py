from pyabc import BOLFI, Prior
import numpy as np
import matplotlib.pyplot as plt

mu0 = 2.5
y0 = np.random.normal(mu0, 1, 2)

prior = Prior('uniform', 0, 5)

def simulator(mu):
    return np.random.normal(mu, 1, 2)

bolfi = BOLFI(priors=[prior], simulator=simulator, observation=y0, summaries=[np.mean], domain=[(0,5)])
thetas = bolfi.sample(threshold=0.5)

plt.hist(thetas, bins=50)
plt.show()
