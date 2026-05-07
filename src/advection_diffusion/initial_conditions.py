import numpy as np
import torch


def gaussian_pulse_np(x, amplitude=1.0, x0=0.5, sigma=0.1):
    x = np.asarray(x)
    return amplitude * np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))


def gaussian_pulse_torch(x, amplitude=1.0, x0=0.5, sigma=0.1):
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for gaussian_pulse_torch.")
    return amplitude * torch.exp(-((x - x0) ** 2) / (2.0 * sigma**2))
