import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import itertools

import pyabc

PLOTS_PER_ROW = 3


def plot_marginals(sampler: pyabc.BaseSampler, plot_all=False, hist_kws={}, kde_kws={}, **kwargs):
    """take a sampler and plot the posterior distribution for all model parameter thetas
    :param sampler: instance of BaseSampler
    :param plot_all: true - plot all thetas for all iterations, false - only plot thetas of last round
    :hist_kws: dictionary of matplotlib properties passed to the hist plot function
    :kde_kws: dictionary of matplotlib properties passed to the kde plot function
    :param kwargs: list of keyword arguments for kde function of scipy.stats
    """
    if sampler.Thetas.shape == (0,):
        raise Warning("Method was called before sampling was done")

    def _plot_thetas(thetas, threshold):
        nonlocal sampler, nr_rows, nr_plots, names, hist_kws, kde_kws, kwargs

        fig = plt.figure()

        # default properties
        if hist_kws.get('bins') is None:
            hist_kws['bins'] = 'auto'

        if hist_kws.get('normed') is None:
            hist_kws['normed'] = True

        if hist_kws.get('alpha') is None:
            hist_kws['alpha'] = 0.4

        if hist_kws.get('edgecolor') is None:
            hist_kws['edgecolor'] = 'k'

        if kwargs.get('color') is None:
            kwargs['color'] = 'darkblue'

        # plot thetas of last iteration
        for plot_id, theta in enumerate(thetas.T):

            if nr_plots < PLOTS_PER_ROW:
                plt.subplot(nr_rows, nr_plots, plot_id + 1)
            else:
                plt.subplot(nr_rows, PLOTS_PER_ROW, plot_id + 1)

            # plot posterior
            plt.hist(theta, **hist_kws)
            # plot KDE and MAP
            # get the bandwidth method argument for scipy
            # and run scipy's kde
            kde = ss.kde.gaussian_kde(theta, bw_method=kde_kws.get('bw_method'), **kde_kws)
            xx = np.linspace(np.min(theta), np.max(theta), 200)
            dens = kde(xx)
            plt.plot(xx, dens, color=kwargs['color'], label="ABC posterior")
            # plot mean
            plt.axvline(np.mean(theta), linewidth=1.2, color=kwargs['color'], linestyle="--", label="mean")
            plt.axvline(xx[np.argmax(dens)], linewidth=1.2, color=kwargs['color'], linestyle=":", label="MAP")

            # label of axis
            if kwargs.get('xlim'):
                plt.xlim(kwargs.get('xlim')[plot_id])
            if kwargs.get('ylim'):
                plt.ylim(kwargs.get('ylim'))
            plt.xlabel(names[plot_id])
            plt.legend(loc="upper right")

        fig.suptitle("ABC Posterior for sampler \n{} with\n".format(
            type(sampler).__name__) + r"$\rho(S(X),S(Y)) < {}, n = {}$".format(
            np.round(threshold, 4),
            sampler.Thetas.shape[0]
        ), y=0.96)

        plt.tight_layout(rect=[0.05, 0, 0.95, 0.85])
        #plt.show()
        return fig

    nr_plots = sampler.Thetas.shape[1]  # number of columns = model parameters
    nr_rows = (nr_plots // (PLOTS_PER_ROW + 1)) + 1  # has to start by one

    names = np.hstack((np.atleast_1d(p.name) for p in sampler.priors))

    if isinstance(sampler, pyabc.BaseSampler):
        fig = _plot_thetas(sampler.Thetas, sampler.threshold)
    else:
        raise TypeError("Type of sampler is unknown.".format(repr(sampler)))

    if isinstance(sampler, pyabc.SMCSampler) & plot_all:
        for epoch, threshold in enumerate(sampler.thresholds):
            xlim = kwargs.get('xlim') or [
                (sampler.particles[0, :, t].min() - 0.1, sampler.particles[0, :, t].max() + 0.1)
                for t
                in
                range(sampler.particles[0].shape[1])]
            _plot_thetas(sampler.particles[epoch], threshold)

    return fig


def plot_pairs(sampler: pyabc.BaseSampler, diagonal='hist', hist_kwds=None, density_kwds=None, range_padding=0.05,
               **kwargs):
    """Plots a scatterplot matrix of subplots.  Each column of "sampler.Thetas" is plotted
    against other columns, resulting in a ncols by ncols grid of subplots with the
    diagonal subplots labeled with "names".  Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containg the subplot grid.
    :param sampler: instance of BaseSampler
    :param diagonal: what to plot on the diagonal (either 'hist', 'kde' or 'names')
    :param hist_kwds: dictionary of kwargs for the call to plt.hist (when diagonal == 'hist')
    :param density_kwds: dictionary of kwargs for the call to plt.plot for kde (when diagonal == 'kde')
    :param range_padding: float, optional
        relative extension of axis range in x and y
        with respect to (x_max - x_min) or (y_max - y_min),
        default 0.05
    :param **kwargs: keyword arguments for the call to plt.scatter on the off-diagonal
    """

    # set defaults for histogram
    hist_kwds = hist_kwds or {}
    if not hist_kwds.get('alpha'):
        hist_kwds['alpha'] = 0.6
    if not hist_kwds.get('edgecolor'):
        hist_kwds['edgecolor'] = 'k'

    density_kwds = density_kwds or {}

    if sampler.Thetas.shape == (0,):
        raise Warning("Method was called before sampling was done")

    names = sampler.priors.names

    # initialize figure and axes
    numdata, numvars = sampler.Thetas.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # compute data limits
    boundaries_list = []
    for theta in sampler.Thetas.T:
        rmin_, rmax_ = np.min(theta), np.max(theta)
        rdelta_ext = (rmax_ - rmin_) * range_padding / 2.
        boundaries_list.append((rmin_ - rdelta_ext, rmax_ + rdelta_ext))

    # iterate all pairs of parameters
    for i, theta_a, name_a in zip(range(numvars), sampler.Thetas.T, names):
        for j, theta_b, name_b in zip(range(numvars), sampler.Thetas.T, names):
            ax = axes[i, j]

            # plot the diagonal
            if i == j:
                if diagonal == 'names':
                    ax.annotate(label, (0.5, 0.5), xycoords='axes fraction',
                                ha='center', va='center')

                elif diagonal == 'hist':
                    ax.hist(theta_a, **hist_kwds)

                elif diagonal in ('kde', 'density'):
                    y = theta_a
                    gkde = ss.gaussian_kde(y)
                    ind = np.linspace(y.min(), y.max(), 1000)
                    ax.plot(ind, gkde.evaluate(ind), **density_kwds)

            # plot the off-diagonal
            else:
                axes[i, j].scatter(theta_b, theta_a, **kwargs)

                ax.set_xlim(boundaries_list[j])
                ax.set_ylim(boundaries_list[i])

            ax.set_xlabel(name_b)
            ax.set_ylabel(name_a)

            if j != 0:
                ax.yaxis.set_visible(False)
            if i != numvars - 1:
                ax.xaxis.set_visible(False)

    # set the axis boundaries and labels
    if numvars > 1:
        lim1 = boundaries_list[0]
        locs = axes[0][1].yaxis.get_majorticklocs()
        locs = locs[(lim1[0] <= locs) & (locs <= lim1[1])]
        adj = (locs - lim1[0]) / (lim1[1] - lim1[0])

        lim0 = axes[0][0].get_ylim()
        adj = adj * (lim0[1] - lim0[0]) + lim0[0]
        axes[0][0].yaxis.set_ticks(adj)

        if np.all(locs == locs.astype(int)):
            # if all ticks are int
            locs = locs.astype(int)
        axes[0][0].yaxis.set_ticklabels(locs)

    fig.suptitle('Scatterplot matrix of parameter posterior for \n {} with\n'.format(
        type(sampler).__name__) + r'$\rho(S(X),S(Y)) < {}, n = {}$'.format(
        np.round(sampler.threshold, 4),
        sampler.Thetas.shape[0]
    ), y=0.96)

    plt.show()
    return fig


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
        return fig
