import numpy as np
from scipy.stats import binom
from scipy.special import erfinv

def uniform_generator(a, b, num_samples=100):
    """
    Generates an array of uniformly distributed random numbers within the specified range.
    """
    np.random.seed(42)
    array = np.random.uniform(a, b, num_samples)
    return array

def inverse_cdf_gaussian(y, mu, sigma):
    """
    Calculates the inverse cumulative distribution function (CDF) of a Gaussian distribution.
    """
    x = (sigma * (np.sqrt(2)) * erfinv(np.dot(2, y) - 1)) + mu
    return x

def gaussian_generator(mu, sigma, num_samples):
    """
    Generates an array of Gaussian distributed random numbers.
    """
    u = uniform_generator(0, 1, num_samples)
    array = inverse_cdf_gaussian(u, mu, sigma)
    return array

def inverse_cdf_binomial(y, n, p):
    """
    Calculates the inverse cumulative distribution function (CDF) of a binomial distribution.
    """
    x = binom.ppf(y, n, p)
    return x

def binomial_generator(n, p, num_samples):
    """
    Generates an array of binomially distributed random numbers.
    """
    u = np.random.uniform(0, 1, num_samples)
    array = inverse_cdf_binomial(u, n, p)
    return array
