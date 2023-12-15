import numpy as np
from scipy.linalg import sqrtm


def calculate_fid(latents1, latents2):
    """ Compute Frechet-Inception Distance between two sets of latents.

    Args:
        latents1: Array of latents with shape [N, H] where N is no. of samples, and H is latent size
        latents2: Array of latents with shape [N, H] where N is no. of samples, and H is latent size

    Returns:
        FID score between two sets of latents.
    """
    # Compute mean and covariance statistics
    mu1 = np.mean(latents1, axis=0)
    sigma1 = np.cov(latents1, rowvar=False)

    mu2 = np.mean(latents2, axis=0)
    sigma2 = np.cov(latents2, rowvar=False)

    # Compute sum squared differences between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)

    # Compute sqrt of product between covariances
    covmean = sqrtm(sigma1.dot(sigma2))

    # Check if imaginary numbers present
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Compute score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid
