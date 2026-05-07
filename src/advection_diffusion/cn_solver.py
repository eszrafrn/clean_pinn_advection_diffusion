from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import factorized

from .initial_conditions import gaussian_pulse_np
from .metrics import compute_mass


@dataclass
class CNConfig:
    L: float = 1.0
    T: float = 0.5
    nx: int = 1000
    nt: int = 1000
    v: float = 0.05
    D: float = 0.01
    bc_type: str = "periodic"  # dirichlet, periodic, zero_flux
    c_left: float = 0.0
    c_right: float = 0.0
    ic_amplitude: float = 1.0
    ic_x0: float = 0.5
    ic_sigma: float = 0.1


class CrankNicolson1D:
    """Crank-Nicolson solver for 1D advection-diffusion.

    - dirichlet menggunakan nodal grid termasuk x=0 and x=L.
    - periodic menggunakan endpoint-free grid.
    - zero_flux menggunakan grid sel konservatif seperti finite volume dengan total flux batas J = v c - D c_x diset ke nol
    """

    def __init__(self, config: CNConfig):
        self.config = config
        self.bc_type = config.bc_type.lower()
        if self.bc_type not in {"dirichlet", "periodic", "zero_flux"}:
            raise ValueError("error: jenis BC yang digunakan tidak tersedia")

        self.L = config.L
        self.T = config.T
        self.nx = config.nx
        self.nt = config.nt
        self.v = config.v
        self.D = config.D
        self.dt = self.T / self.nt

        if self.bc_type == "dirichlet":
            self.dx = self.L / (self.nx - 1)
            self.x = np.linspace(0.0, self.L, self.nx)
            self.grid_type = "nodal"
        elif self.bc_type == "periodic":
            self.dx = self.L / self.nx
            self.x = np.linspace(0.0, self.L, self.nx, endpoint=False)
            self.grid_type = "periodic"
        else:
            self.dx = self.L / self.nx
            self.x = (np.arange(self.nx) + 0.5) * self.dx
            self.grid_type = "cell"

        self.t = np.linspace(0.0, self.T, self.nt + 1)
        self.operator = self._build_spatial_operator()
        self.lhs, self.rhs = self._build_cn_matrices()

        self.solve_lhs = factorized(csc_matrix(self.lhs))
        #self.solve_lhs = lambda b: np.linalg.solve(self.lhs, b)

    @property
    def peclet(self):
        return np.inf if self.D == 0.0 else self.v * self.L / self.D

    def initial_condition(self):
        cfg = self.config
        c = gaussian_pulse_np(
            self.x,
            amplitude=cfg.ic_amplitude,
            x0=cfg.ic_x0,
            sigma=cfg.ic_sigma,
        )
        if self.bc_type == "dirichlet":
            c[0] = cfg.c_left
            c[-1] = cfg.c_right
        return c.astype(float)

    def _build_spatial_operator(self):
        n = self.nx
        dx = self.dx
        v = self.v
        D = self.D
        A = np.zeros((n, n), dtype=float)

        lower = v / (2.0 * dx) + D / dx**2
        center = -2.0 * D / dx**2
        upper = -v / (2.0 * dx) + D / dx**2

        if self.bc_type == "dirichlet":
            for i in range(1, n - 1):
                A[i, i - 1] = lower
                A[i, i] = center
                A[i, i + 1] = upper
            return A

        if self.bc_type == "periodic":
            for i in range(n):
                A[i, (i - 1) % n] = lower
                A[i, i] = center
                A[i, (i + 1) % n] = upper
            return A

        # zero total flux J = v c - D c_x = 0 at the boundary faces.
        for i in range(1, n - 1):
            A[i, i - 1] = lower
            A[i, i] = center
            A[i, i + 1] = upper

        A[0, 0] = -v / (2.0 * dx) - D / dx**2
        A[0, 1] = -v / (2.0 * dx) + D / dx**2

        A[n - 1, n - 2] = v / (2.0 * dx) + D / dx**2
        A[n - 1, n - 1] = v / (2.0 * dx) - D / dx**2
        return A

    def _build_cn_matrices(self):
        n = self.nx
        I = eye(n, format="csc")
        #I = np.eye(n)
        lhs = I - 0.5 * self.dt * self.operator
        rhs = I + 0.5 * self.dt * self.operator

        if self.bc_type == "dirichlet":
            lhs[0, :] = 0.0
            lhs[0, 0] = 1.0
            lhs[-1, :] = 0.0
            lhs[-1, -1] = 1.0
            rhs[0, :] = 0.0
            rhs[-1, :] = 0.0

        return csc_matrix(lhs), csc_matrix(rhs)
        #return lhs, rhs

    def step(self, c):
        b = self.rhs @ c
        if self.bc_type == "dirichlet":
            b[0] = self.config.c_left
            b[-1] = self.config.c_right
        c_next = self.solve_lhs(b)
        if self.bc_type == "dirichlet":
            c_next[0] = self.config.c_left
            c_next[-1] = self.config.c_right
        return np.asarray(c_next, dtype=float)

    def solve(self, save_history=True):
        c = self.initial_condition()
        mass_history = [compute_mass(c, x=self.x, L=self.L, grid_type=self.grid_type)]
        history = [c.copy()] if save_history else None

        for _ in range(self.nt):
            c = self.step(c)
            mass_history.append(
                compute_mass(c, x=self.x, L=self.L, grid_type=self.grid_type)
            )
            if save_history:
                history.append(c.copy())

        result = {
            "x": self.x,
            "t": self.t,
            "c_final": c,
            "mass": np.asarray(mass_history),
            "grid_type": self.grid_type,
        }
        if save_history:
            result["c"] = np.asarray(history)
        return result
