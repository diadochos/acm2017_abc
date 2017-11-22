from pyabc import Prior
import numpy as np

mu = Prior('multivariate_gaussian', np.array([0,0]), np.diag([1,1]))

priors2d = [mu]*2

print(np.vstack([mu.sample(size=10) for mu in priors2d]).shape)


m = Prior('gaussian', 0, 1)

priors1d = [m]*2
print(np.vstack([m.sample(size=10) for m in priors1d]).shape)
