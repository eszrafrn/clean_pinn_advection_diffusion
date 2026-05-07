import numpy as np
import torch

from scipy.stats import qmc


def to_tensor(array, device="cpu", requires_grad=False):
    tensor = torch.tensor(array, dtype=torch.float32, device=device)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def sample_collocation(n_points, L, T, seed=None):
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    sample = sampler.random(n=n_points)
    #rng = np.random.default_rng(seed)
    #sample = rng.random((n_points, 2))

    x = L * sample[:, 0:1]
    t = T * sample[:, 1:2]
    return x, t


def sample_initial(n_points, L, seed=None):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, L, size=(n_points, 1))
    t = np.zeros_like(x)
    return x, t


def sample_boundary_times(n_points, T, seed=None):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, T, size=(n_points, 1))


def prepare_training_data(n_r, n_ic, n_bc, L, T, bc_type, device="cpu", seed=42):
    bc_type = bc_type.lower()
    x_r, t_r = sample_collocation(n_r, L, T, seed=seed)
    x_ic, t_ic = sample_initial(n_ic, L, seed=seed + 1)
    t_bc = sample_boundary_times(n_bc, T, seed=seed + 2)

    data = {
        "x_r": to_tensor(x_r, device=device, requires_grad=True),
        "t_r": to_tensor(t_r, device=device, requires_grad=True),
        "x_ic": to_tensor(x_ic, device=device, requires_grad=False),
        "t_ic": to_tensor(t_ic, device=device, requires_grad=False),
        "t_bc": to_tensor(t_bc, device=device, requires_grad=True),
    }

    if bc_type == "dirichlet":
        rng = np.random.default_rng(seed + 3)
        x_bc = rng.choice([0.0, L], size=(n_bc, 1))
        data["x_bc"] = to_tensor(x_bc, device=device, requires_grad=False)
    elif bc_type in {"periodic", "zero_flux"}:
        data["x_left"] = to_tensor(np.zeros_like(t_bc), device=device, requires_grad=True)
        data["x_right"] = to_tensor(np.full_like(t_bc, L), device=device, requires_grad=True)
    else:
        raise ValueError("bc_type error, bc yang tersedia: dirichlet, periodic, or zero_flux")

    return data
