import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import pyabc

PLOTS_PER_ROW = 3

def plot_marginals(sampler: pyabc.BaseSampler, plot_particles=False, as_circle=True, kde=False, **kwargs):
    """take a sampler and plot the posterior distribution for all model parameter thetas
    :param sampler: instance of BaseSampler
    :param plot_particles: true - plot particels for all iterations, false - otherwise
    :param as_circle: true - plot particles as circles on x-axis, otherwise as scatter plot
    :param kde: true - use kernel density estimation function of scipy.stats to draw posterior distribution
    :param kwargs: list of keyword arguments for kde function of scipy.stats
    """
    if sampler.Thetas.shape == (0,):
        raise Warning("Method was called before sampling was done")

    def _plot_thetas(thetas, threshold):
        nonlocal sampler, kde, nr_rows, names, kwargs

        fig = plt.figure()

        # plot thetas of last iteration
        for plot_id, theta in enumerate(thetas.T):
            plt.subplot(nr_rows, PLOTS_PER_ROW, plot_id + 1)

            # plot posterior
            plt.hist(theta, edgecolor="k", bins='auto', normed=True, alpha=0.4)
            # plot mean
            plt.axvline(np.mean(theta), linewidth=1.2, color="m", linestyle="--", label="mean")
            # plot MAP
            if kde:
                # get the bandwidth method argument for scipy
                # and run scipy's kde
                kde = ss.kde.gaussian_kde(theta, bw_method=kwargs.get('bw_method'))
                xx = np.linspace(np.min(theta) - 0.1, np.max(thetas) + 0.1, 200)
                dens = kde(xx)
                plt.plot(xx, dens)
                plt.axvline(xx[np.argmax(dens)], linewidth=1.2, color="m", linestyle=":", label="MAP")

            # label of axis
            plt.xlabel(names[plot_id])
            plt.legend(loc="upper right")

        fig.suptitle("Posterior for all model parameters with\n" + r"$\rho(S(X),S(Y)) < {}, n = {}$".format(
            threshold,
            sampler.Thetas.shape[0]
        ), y=0.96)

        plt.tight_layout(rect=[0.05, 0, 0.95, 0.85])
        plt.show()

    def _plot_particles(particles, weights, threshold, as_circle=True):
        nonlocal sampler, kde, nr_rows, names, kwargs

        #norm weights
        if weights.max() != weights.min():
            weights = np.divide(weights - weights.min(), weights.max() - weights.min()) / 100

        for plot_id, theta in enumerate(particles.T):

            fig = plt.figure()

            # draw for each theta a circle with radius equal to its weight
            if as_circle:
                for i, x in enumerate(theta):
                    circle = plt.Circle((x, 0), radius=weights[i], alpha=0.4)
                    plt.gca().add_patch(circle)

                plt.ylabel("weight $w_{{i,t}}$")
                plt.xlabel(names[plot_id])
                plt.xlim([particles.min() - 0.1, particles.max() + 0.1])
                plt.ylim([-weights.max(), weights.max()])
                plt.axis("equal")

            else:
                plt.plot(theta, weights, "o", alpha=0.4)
                plt.ylabel("weight $w_{{i,t}}$")

                plt.xlabel(names[plot_id])
                plt.xlim([particles.min() - 0.1, particles.max() + 0.1])
                plt.axis("equal")

        plt.title("Distribution of Particles represented by their weights\n" + r"$\rho(S(X),S(Y)) < {}, n = {}$".format(
            threshold, weights.shape[0]))

        plt.tight_layout(rect=[0.05, 0, 0.95, 0.85])
        plt.show()

    nr_plots = sampler.Thetas.shape[1]  # number of columns = model parameters
    nr_rows = (nr_plots // (PLOTS_PER_ROW + 1)) + 1  # has to start by one

    names = np.hstack((np.atleast_1d(p.name) for p in sampler.priors))

    if isinstance(sampler, pyabc.RejectionSampler):

        _plot_thetas(sampler.Thetas, sampler.threshold)

    elif isinstance(sampler, pyabc.SMCSampler):
        if plot_particles:
            for epoch, threshold in enumerate(sampler.thresholds):
                _plot_thetas(sampler.particles[epoch], threshold)
                _plot_particles(sampler.particles[epoch], sampler.weights[epoch], threshold, as_circle)
        else:
            _plot_thetas(sampler.Thetas, sampler.thresholds[-1])

    else:
        raise TypeError("Type of sampler is unknown.".format(repr(sampler)))
