# Approximate Bayesian Computation

Most Bayesian inference methods rely on the ability to calculate the likelihood, i.e. the probability of the data given the model parameters. For many domains, this likelihood function is intractable or unknown, but one can specify a (stochastic) simulator function. Approximate Bayesian Computation (ABC) methods (also called Likelihood Free Inference) tackle this problem by circumventing calculation of the likelihood.

This package containt implementation of basic ABC algorithms and standard toy examples for ABC problems.

The `pyabc` package provides the following sampling-based ABC algorithms:

- [Rejection ABC](http://www.genetics.org/content/genetics/162/4/2025.full.pdf)
- [MCMC ABC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC307566/?tool=pmcentrez&report=abstract)
- [SMC ABC](http://www.pnas.org/cgi/doi/10.1073/pnas.0607208104)
- [ABCDE](https://www.sciencedirect.com/science/article/pii/S0022249612000752?via%3Dihub)

In addition, it implements a more recent approach called [Bayesian Optimization for Likelihood Free Inference (BOLFI)](http://arxiv.org/abs/1501.03291).

There are also first attempts at implementing a Regression ABC method called [Bayesian Conditional Density Estimation Using Mixture Density Networks](http://arxiv.org/abs/1605.06376) methods in PyTorch (see this [notebood](https://github.com/compercept/acm2017_abc/blob/master/notebooks/RegressionABC.ipynb)).

We also implemented a framework for toy problems that provides access to simulator functions and summary statistics and is easy to new problems. The following examples from the ABC literature are already implemented:
- [Mixture of Gaussians](http://www.pnas.org/cgi/doi/10.1073/pnas.0607208104)
- [Ricker model](http://www.nature.com/doifinder/10.1038/nature09319)
- [Tuberculosis spread](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1526704/)
- [Episodic Memory](https://www.sciencedirect.com/science/article/pii/S0022249612000272)

For an introduction on how to perform inference with the `pyabc` package, see the [HowTo](https://github.com/compercept/acm2017_abc/blob/master/HowTo.ipynb) notebook.
