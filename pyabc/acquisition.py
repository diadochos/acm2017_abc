from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.optimization import acquisition_optimizer

class MaxPosteriorVariance(AcquisitionBase):

    analytical_gradient_prediction = True

    def __init__(self, model, space, prior, quantile_eps=.01, optimizer=None):

        optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)
        super(MaxPosteriorVariance, self).__init__(model, space, optimizer)

        self.prior = prior
        self.quantile_eps = quantile_eps
        # The ABC threshold is initialised to a pre-set value as the gp is not yet fit.
        self.eps = .1



    def acquisition_function(self,x):
        mean, var = self.model.predict(theta_new, noiseless=True)
        sigma2_n = self.model.noise

        # Using the cdf of Skewnorm to avoid explicit Owen's T computation.
        a = np.sqrt(sigma2_n) / np.sqrt(sigma2_n + 2. * var)  # Skewness.
        scale = np.sqrt(sigma2_n + var)
        phi_skew = ss.skewnorm.cdf(self.eps, a, loc=mean, scale=scale)
        phi_norm = ss.norm.cdf(self.eps, loc=mean, scale=scale)
        var_p_a = phi_skew - phi_norm**2

        val_prior = self.prior.pdf(theta_new).ravel()[:, np.newaxis]

        var_approx_posterior = val_prior**2 * var_p_a
        return var_approx_posterior

    def acquisition_function_withGradients(self,x):
        phi = ss.norm.cdf
        mean, var = self.model.predict(theta_new, noiseless=True)
        grad_mean, grad_var = self.model.predictive_gradients(theta_new)
        sigma2_n = self.model.noise
        scale = np.sqrt(sigma2_n + var)

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
        term_prior = self.prior.pdf(theta_new).ravel()[:, np.newaxis]
        grad_prior_log = self.prior.gradient_logpdf(theta_new)
        term_grad_prior = term_prior * grad_prior_log

        gradient = 2. * term_prior * (int_1 - int_2) * term_grad_prior + \
            term_prior**2 * (grad_int_1 - grad_int_2)
        return gradient
