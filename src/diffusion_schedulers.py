"""Diffusion Schedulers"""

import numpy as np


def linear(beta_min: float, beta_max: float, diffusion_times: list[float]) -> list[float]:
    """
    Noise variance as input and returns the alpha bar values at different time steps
    based on a linearly increasing schedule.

    Args:
        beta_min (float): Minimum noise variance.
        beta_max (float): Maximum noise variance.
        diffusion_times (list[float]): List of diffusion times (normalized between 0 and 1).
    Returns:
        list[float]: List of alpha bar values at different time steps,

    """
    betas = beta_min + diffusion_times * (beta_max - beta_min)
    alphas = 1 - betas
    return np.cumprod(alphas)


def quadratic(beta_min: float, beta_max: float, diffusion_times: list[float]) -> list[float]:
    """
    Returns the alpha bar values at different time steps based on a quadratic increasing schedule
    Args:
        beta_min (float): Minimum noise variance.
        beta_max (float): Maximum noise variance.
        diffusion_times (list[float]): List of diffusion times (normalized between 0 and 1).
    Returns:
        list[float]: List of alpha bar values at different time steps,

    """
    betas = beta_min + (diffusion_times**2) * (beta_max - beta_min)
    alphas = 1 - betas
    return np.cumprod(alphas)


def cosine(diffusion_times: list[float], offset: float = 0.0) -> list[float]:
    """
    Returns the alpha bar values at different time steps using sinusoid schedule.
    Args:
        diffusion_times (list[float]): List of diffusion times (normalized between 0 and 1).
        offset (float): Offset to apply to the diffusion times.
    Returns:
        list[float]: List of alpha bar values at different time steps,
    """
    diffusion_times = (np.array(diffusion_times) + offset) / (1 + offset)
    return np.cos(diffusion_times * (np.pi / 2)) ** 2


def square_root(diffusion_times: list[float], offset: float) -> list[float]:
    """
    Returns the alpha bar values at different time steps using a square root schedule.
    Args:
        diffusion_times (list[float]): List of diffusion times (normalized between 0 and 1).
        offset (float): Offset to apply to the diffusion times.
    Returns:
        list[float]: List of alpha bar values at different time steps,
    """
    return 1 - (diffusion_times + offset) ** 0.5
