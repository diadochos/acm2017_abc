import GPyOpt
from GPyOpt.acquisitions.base import AcquisitionBase
import scipy.stats as ss
import numpy as np


class MaxPosteriorVariance(AcquisitionBase):

    def __init__(self, model, space, prior, eps=.01, optimizer=None):

        super(MaxPosteriorVariance, self).__init__(model, space, optimizer)

        self.prior = prior
        self.eps = eps

    def _compute_acq(self, x):
        # print('Compute acq')
        # print(x.shape)
        mean, var = self.model.model.predict_noiseless(x)
        sigma2_n = self.model.model.Gaussian_noise.variance[0]

        # Using the cdf of Skewnorm to avoid explicit Owen's T computation.
        a = np.sqrt(sigma2_n) / np.sqrt(sigma2_n + 2. * var)  # Skewness.
        scale = np.sqrt(sigma2_n + var)
        phi_skew = ss.skewnorm.cdf(self.eps, a, loc=mean, scale=scale)
        phi_norm = ss.norm.cdf(self.eps, loc=mean, scale=scale)
        var_p_a = phi_skew - phi_norm**2

        val_prior = self.prior.pdf(x).ravel()[:, np.newaxis]

        var_approx_posterior = val_prior**2 * var_p_a
        return -var_approx_posterior

    def _compute_acq_withGradients(self, x):
        phi = ss.norm.cdf
        mean, var = self.model.model.predict_noiseless(x)
        grad_mean, grad_var = self.model.model.predictive_gradients(x)
        sigma2_n = self.model.model.Gaussian_noise.variance[0]
        scale = np.sqrt(sigma2_n + var)

        # Using the cdf of Skewnorm to avoid explicit Owen's T computation.
        a = np.sqrt(sigma2_n) / np.sqrt(sigma2_n + 2. * var)  # Skewness.
        scale = np.sqrt(sigma2_n + var)
        phi_skew = ss.skewnorm.cdf(self.eps, a, loc=mean, scale=scale)
        phi_norm = ss.norm.cdf(self.eps, loc=mean, scale=scale)
        var_p_a = phi_skew - phi_norm**2

        val_prior = self.prior.pdf(x).ravel()[:, np.newaxis]

        var_approx_posterior = val_prior**2 * var_p_a

        a = (self.eps - mean) / scale
        b = np.sqrt(sigma2_n) / np.sqrt(sigma2_n + 2 * var)
        grad_a = (-1. / scale) * grad_mean - \
            ((self.eps - mean) / (2. * (sigma2_n + var)**(1.5))) * grad_var
        grad_b = (-np.sqrt(sigma2_n) / (sigma2_n + 2 * var)**(1.5)) * grad_var

        _phi_a = phi(a)
        int_1 = _phi_a - _phi_a**2
        int_2 = phi(self.eps, loc=mean, scale=scale) \
            - ss.skewnorm.cdf(self.eps, b, loc=mean, scale=scale)
        grad_int_1 = (1. - 2 * _phi_a) * \
            (np.exp(-.5 * (a**2)) / np.sqrt(2. * np.pi)) * grad_a
        grad_int_2 = (1. / np.pi) * \
            (((np.exp(-.5 * (a**2) * (1. + b**2))) / (1. + b**2)) * grad_b +
                (np.sqrt(np.pi / 2.) * np.exp(-.5 * (a**2)) * (1. - 2. * phi(a * b)) * grad_a))

        # Obtaining the gradient prior by applying the following rule:
        # (log f(x))' = f'(x)/f(x) => f'(x) = (log f(x))' * f(x)
        term_prior = self.prior.pdf(x).ravel()[:, np.newaxis]
        grad_prior_log = self.prior.gradient_logpdf(x)
        term_grad_prior = term_prior * grad_prior_log

        gradient = 2. * term_prior * (int_1 - int_2) * term_grad_prior + \
            term_prior**2 * (grad_int_1 - grad_int_2)
        return -var_approx_posterior, -gradient

    @staticmethod
    def fromDict(model, space, optimizer, cost_withGradients, config):
        raise NotImplementedError()
