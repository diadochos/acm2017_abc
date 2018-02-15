import torch
import torch.nn as nn
import numpy as np
import scipy.stats as ss

class MDN(nn.Module):
    def __init__(self, hidden_size, num_mixtures, input_dim):
        super(MDN, self).__init__()
        self.fc_in = nn.Linear(input_dim, hidden_size) 
        self.tanh = nn.Tanh()
        self.pi_out = torch.nn.Sequential(
              nn.Linear(hidden_size, num_mixtures),
              nn.Softmax()
            )
        self.sigma_out = nn.Linear(hidden_size, num_mixtures)
        self.mu_out = nn.Linear(hidden_size, num_mixtures)  

    def forward(self, x):
        out = self.fc_in(x)
        out = self.tanh(out)
        out_pi = self.pi_out(out)
        out_sigma = torch.exp(self.sigma_out(out))
        out_mu = self.mu_out(out)
        return (out_pi, out_sigma, out_mu)
    
    def pdf(self, x, theta):
        (out_pi_test, out_sigma_test, out_mu_test) = self(x)

        out_pi = out_pi_test.data.numpy().T
        out_sigma = out_sigma_test.data.numpy().T
        out_mu = out_mu_test.data.numpy().T

        pdf = np.array([ss.norm.pdf(theta, mu, sigma) * pi for mu, sigma, pi in zip(out_mu, out_sigma, out_pi)])
        return pdf.sum(axis=0)
    
    def sample(x, n):
        (out_pi_test, out_sigma_test, out_mu_test) = model(x)

        out_pi = out_pi_test.data.numpy().T
        out_sigma = out_sigma_test.data.numpy().T
        out_mu = out_mu_test.data.numpy().T

        data = np.zeros((n, out_pi.size))

        num_distr= out_pi.size

        for idx, distr in enumerate(zip(out_mu, out_sigma)):
            mu, sigma = distr
            data[:, idx] = np.random.normal(mu, sigma, size=n)
        random_idx = np.random.choice(np.arange(num_distr), size=(n,), p=out_pi[:,0])
        sample = data[np.arange(n), random_idx]
        return sample
        
    
    
oneDivSqrtTwoPI = 1.0 / np.sqrt(2.0*np.pi) # normalisation factor for gaussian.
def gaussian_distribution(theta, mu, sigma):
    # braodcast subtraction with mean and normalization to sigma
    result = (theta.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = - 0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * oneDivSqrtTwoPI

def mdn_loss_function(out_pi, out_sigma, out_mu, theta):
    result = gaussian_distribution(theta, out_mu, out_sigma) * out_pi
    result = torch.sum(result, dim=1)
    result = - torch.log(result)
    return torch.mean(result)

