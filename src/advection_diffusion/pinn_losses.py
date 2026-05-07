import numpy as np
import torch
import torch.nn.functional as F

from .initial_conditions import gaussian_pulse_torch


def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def predict_c(model, x, t):
    return model(torch.cat([x, t], dim=1))


def pde_residual(model, x, t, v, D):
    c = predict_c(model, x, t)
    c_t = grad(c, t)
    c_x = grad(c, x)
    c_xx = grad(c_x, x)
    return c_t + v * c_x - D * c_xx


def total_flux(model, x, t, v, D):
    c = predict_c(model, x, t)
    c_x = grad(c, x)
    return v * c - D * c_x


def loss_ic(model, data, ic_params):
    c_true = gaussian_pulse_torch(data["x_ic"], **ic_params)
    c_pred = predict_c(model, data["x_ic"], data["t_ic"])
    return F.mse_loss(c_pred, c_true)


def loss_bc_dirichlet(model, data, c_left=0.0, c_right=0.0, L=1.0):
    x_bc = data["x_bc"]
    t_bc = data["t_bc"]
    target = torch.where(
        torch.isclose(x_bc, torch.zeros_like(x_bc)),
        torch.full_like(x_bc, float(c_left)),
        torch.full_like(x_bc, float(c_right)),
    )
    return F.mse_loss(predict_c(model, x_bc, t_bc), target)


def loss_bc_periodic(model, data, v, D):
    x_left = data["x_left"]
    x_right = data["x_right"]
    t_bc = data["t_bc"]
    c_left = predict_c(model, x_left, t_bc)
    c_right = predict_c(model, x_right, t_bc)
    j_left = total_flux(model, x_left, t_bc, v, D)
    j_right = total_flux(model, x_right, t_bc, v, D)
    return F.mse_loss(c_left, c_right) + F.mse_loss(j_left, j_right)


def loss_bc_zero_flux(model, data, v, D):
    x_left = data["x_left"]
    x_right = data["x_right"]
    t_bc = data["t_bc"]
    j_left = total_flux(model, x_left, t_bc, v, D)
    j_right = total_flux(model, x_right, t_bc, v, D)
    return F.mse_loss(j_left, torch.zeros_like(j_left)) + F.mse_loss(
        j_right, torch.zeros_like(j_right)
    )


def gaussian_quadrature(n=80, L=1.0, device="cpu"):
    x_gl, w_gl = np.polynomial.legendre.leggauss(n)
    x = 0.5 * (x_gl + 1.0) * L
    w = 0.5 * w_gl * L
    return (
        torch.tensor(x, dtype=torch.float32, device=device).reshape(-1, 1),
        torch.tensor(w, dtype=torch.float32, device=device).reshape(-1, 1),
    )


def mass_at_times(model, x_quad, w_quad, t_samples):
    masses = []
    for i in range(t_samples.shape[0]):
        t = t_samples[i : i + 1].repeat(x_quad.shape[0], 1)
        c = predict_c(model, x_quad, t)
        masses.append(torch.sum(c * w_quad))
    return torch.stack(masses)


def loss_mass_penalty(model, x_quad, w_quad, t_samples, M0):
    masses = mass_at_times(model, x_quad, w_quad, t_samples)
    target = torch.full_like(masses, float(M0))
    return F.mse_loss(masses, target)


def total_pinn_loss(model, data, params, mass_data=None):
    bc_type = params["bc_type"].lower()
    v = params["v"]
    D = params["D"]
    ic_params = params["ic_params"]

    L_ic = loss_ic(model, data, ic_params)
    residual = pde_residual(model, data["x_r"], data["t_r"], v, D)
    L_pde = F.mse_loss(residual, torch.zeros_like(residual))

    if bc_type == "dirichlet":
        L_bc = loss_bc_dirichlet(
            model,
            data,
            c_left=params.get("c_left", 0.0),
            c_right=params.get("c_right", 0.0),
            L=params.get("L", 1.0),
        )
    elif bc_type == "periodic":
        L_bc = loss_bc_periodic(model, data, v, D)
    elif bc_type == "zero_flux":
        L_bc = loss_bc_zero_flux(model, data, v, D)
    else:
        raise ValueError("bc_type must be: dirichlet, periodic, or zero_flux")

    total = (
        params.get("lambda_ic", 100.0) * L_ic
        + params.get("lambda_bc", 10.0) * L_bc
        + params.get("lambda_pde", 1.0) * L_pde
    )

    L_mass = torch.tensor(0.0, dtype=torch.float32, device=data["x_r"].device)
    if mass_data is not None and params.get("lambda_mass", 0.0) > 0.0:
        L_mass = loss_mass_penalty(
            model,
            mass_data["x_quad"],
            mass_data["w_quad"],
            mass_data["t_samples"],
            mass_data["M0"],
        )
        total = total + params["lambda_mass"] * L_mass

    return total, {
        "L_ic": L_ic,
        "L_bc": L_bc,
        "L_pde": L_pde,
        "L_mass": L_mass,
    }

