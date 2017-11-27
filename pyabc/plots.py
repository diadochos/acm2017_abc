import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import pyabc

PLOTS_PER_ROW = 3

def plot_marginals(sampler: pyabc.BaseSampler, plot_all=False, kde=True, **kwargs):
    """take a sampler and plot the posterior distribution for all model parameter thetas
    :param sampler: instance of BaseSampler
    :param plot_all: true - plot all thetas for all iterations, false - only plot thetas of last round
    :param kde: true - use kernel density estimation function of scipy.stats to draw posterior distribution
    :param kwargs: list of keyword arguments for kde function of scipy.stats
    """
    if sampler.Thetas.shape == (0,):
        raise Warning("Method was called before sampling was done")

    def _plot_thetas(thetas, threshold, xlim=None):
        nonlocal sampler, kde, nr_rows, names, kwargs

        fig = plt.figure()

        # plot thetas of last iteration
        for plot_id, theta in enumerate(thetas.T):
            plt.subplot(nr_rows, PLOTS_PER_ROW, plot_id + 1)

            # plot posterior
            plt.hist(theta, edgecolor="k", bins='auto', normed=kwargs.get('normed'), alpha=0.4)
            # plot mean
            plt.axvline(np.mean(theta), linewidth=1.2, color="m", linestyle="--", label="mean")
            # plot MAP
            if kde:
                # get the bandwidth method argument for scipy
                # and run scipy's kde
                kde = ss.kde.gaussian_kde(theta, bw_method=kwargs.get('bw_method'))
                xx = np.linspace(np.min(theta), np.max(theta), 200)
                dens = kde(xx)
                plt.plot(xx, dens)
                plt.axvline(xx[np.argmax(dens)], linewidth=1.2, color="m", linestyle=":", label="MAP")

            # label of axis
            if xlim:
                plt.xlim(xlim)
            plt.xlabel(names[plot_id])
            plt.legend(loc="upper right")

        fig.suptitle("Posterior for all model parameters with\n" + r"$\rho(S(X),S(Y)) < {}, n = {}$".format(
            threshold,
            sampler.Thetas.shape[0]
        ), y=0.96)

        plt.tight_layout(rect=[0.05, 0, 0.95, 0.85])
        plt.show()


    nr_plots = sampler.Thetas.shape[1]  # number of columns = model parameters
    nr_rows = (nr_plots // (PLOTS_PER_ROW + 1)) + 1  # has to start by one

    names = np.hstack((np.atleast_1d(p.name) for p in sampler.priors))

    if isinstance(sampler, pyabc.BaseSampler):

        _plot_thetas(sampler.Thetas, sampler.threshold)

    elif isinstance(sampler, pyabc.SMCSampler) & plot_all:
        for epoch, threshold in enumerate(sampler.thresholds):
            xlim = (sampler.particles[0].min() - 0.1, sampler.particles[0].max() + 0.1)
            _plot_thetas(sampler.particles[epoch], threshold, xlim)
    else:
        raise TypeError("Type of sampler is unknown.".format(repr(sampler)))


def plot_particles(sampler: pyabc.BaseSampler, as_circles=True, equal_axes=True, **kwargs):
    """take a sampler and plot the particles for each iteration as vertical circle plot

    :param sampler: instance of BaseSampler
    :param as_circles: plot particles as circles with their weights as radius
    :param equal_axes: make axes of plot equally scaled
    :return:
    """
    if sampler.Thetas.shape == (0,):
        raise Warning("Method was called before sampling was done")

    if not isinstance(sampler, pyabc.SMCSampler):
        raise TypeError("Type of sampler is unknown.".format(repr(sampler)))

    nr_epochs, nr_samples, nr_thetas = sampler.particles.shape
    delta_max_y = sampler.particles[0].max() - sampler.particles[0].min()
    CIRCLE_MAX_RADIUS = delta_max_y / 50
    y_lim = (sampler.particles[0].min() - 2 * CIRCLE_MAX_RADIUS, sampler.particles[0].max() + 2 * CIRCLE_MAX_RADIUS)


    names = np.hstack((np.atleast_1d(p.name) for p in sampler.priors))

    # plot for each model parameter theta
    for param_id in range(nr_thetas):

        fig = plt.figure()

        name = names[param_id]
        xticks = []
        # plot all particles for a given theta
        for epoch in range(nr_epochs):

            weights = sampler.weights[epoch]

            # norm weights
            if weights.max() != weights.min():
                weights = np.divide(weights - weights.min(),
                                    weights.max() - weights.min())
            else:
                weights = np.sqrt(weights / weights[0])

            # will be used as radius
            weights *= CIRCLE_MAX_RADIUS

            yy = (sampler.particles[epoch].T)[param_id]
            delta_x = CIRCLE_MAX_RADIUS + epoch * 10 * CIRCLE_MAX_RADIUS
            xticks.append(delta_x)

            # draw for each theta a circle with radius equal to its weight
            if as_circles:
                for i, y in enumerate(yy):
                    circle = plt.Circle(
                        (delta_x, y),
                        radius=weights[i],
                        facecolor="C{}".format(epoch),
                        alpha=kwargs.get("alpha")
                    )
                    plt.gca().add_patch(circle)

            else:
                plt.plot(np.repeat(delta_x, nr_samples), yy, "o", alpha=kwargs.get("alpha"))

            # plot mean
            if epoch == 0:
                plt.plot(
                    [delta_x - CIRCLE_MAX_RADIUS, delta_x + CIRCLE_MAX_RADIUS],
                    [np.mean((sampler.particles[epoch].T)[param_id])] * 2,
                    linewidth=1.2,
                    color="m",
                    linestyle="--",
                    label="mean"
                )
            else:
                plt.plot(
                    [delta_x - CIRCLE_MAX_RADIUS, delta_x + CIRCLE_MAX_RADIUS],
                    [np.mean((sampler.particles[epoch].T)[param_id])] * 2,
                    linewidth=1.2,
                    color="m",
                    linestyle="--"
                )

            plt.ylabel("weights $w_{{i,t}}$ for {}".format(name))
            plt.xlabel("epochs")
            plt.xticks(xticks, [str(x) for x in range(1, nr_epochs + 1)])
            plt.ylim(y_lim)
            plt.xlim((xticks[0] - CIRCLE_MAX_RADIUS, xticks[-1] + CIRCLE_MAX_RADIUS))
            if equal_axes:
                plt.axis("equal")

        plt.title("Distribution of Particles represented by their weights\n" + r"$\epsilon \in {}, n = {}$".format(
            sampler.thresholds, weights.shape[0]))
        plt.legend(loc="upper right")
        plt.show()