import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# plot a multivariate normal contour
def plot_mvnormal(mean=np.array([0.0, 0.0]), cov=np.array([[1,0.0],[0.0,1]]), ax=None, **kwargs):
    
    if not ax:
        f, ax = plt.subplots()
    else:
        f = ax.get_figure()
    
    delta = 0.025
    x = np.arange(0, 10, delta)
    y = np.arange(0, 10, delta)
    X, Y = np.meshgrid(x, y)
    Z = mlab.bivariate_normal(X, Y, mux=mean[0], muy=mean[1], 
                               sigmax=np.sqrt(cov[0,0]), sigmay=np.sqrt(cov[1,1]), 
                               sigmaxy=cov[0,1])

    ax.contour(X, Y, Z, **kwargs)
    ax.set_xlim((0,10))
    ax.set_ylim((0,10))
    ax.axis('equal')
    
    return f, ax