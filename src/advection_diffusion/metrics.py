import numpy as np


def compute_mass(c, x=None, L=1.0, grid_type="nodal"):
    """Compute total mass.

    grid_type:
      - "nodal": values include both x=0 and x=L, use trapezoid.
      - "periodic" or "cell": values represent cells / endpoint-free periodic grid.
    """
    c = np.asarray(c, dtype=float).reshape(-1)
    if grid_type in {"periodic", "cell"}:
        dx = L / c.size
        return float(np.sum(c) * dx)

    if x is None:
        x = np.linspace(0.0, L, c.size)
    x = np.asarray(x, dtype=float).reshape(-1)
    return float(np.trapezoid(c, x=x))


def relative_l2_error(pred, ref):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    denom = np.linalg.norm(ref.reshape(-1))
    if denom < 1e-14:
        return 0.0 if np.linalg.norm(pred.reshape(-1)) < 1e-14 else np.inf
    return float(np.linalg.norm((pred - ref).reshape(-1)) / denom)


def linf_error(pred, ref):
    pred = np.asarray(pred, dtype=float)
    ref = np.asarray(ref, dtype=float)
    return float(np.max(np.abs(pred - ref)))


def conservation_error(mass, initial_mass):
    return float(abs(mass - initial_mass) / max(abs(initial_mass), 1e-14))


def reference_mass_error(mass_pred, mass_ref, initial_mass):
    return float(abs(mass_pred - mass_ref) / max(abs(initial_mass), 1e-14))


def total_variation(c, periodic=False):
    c = np.asarray(c, dtype=float).reshape(-1)
    tv = float(np.sum(np.abs(np.diff(c))))
    if periodic and c.size > 1:
        tv += float(abs(c[0] - c[-1]))
    return tv


def first_threshold_time(t, values, threshold):
    values = np.asarray(values)
    mask = values > threshold
    if not np.any(mask):
        return None
    return float(np.asarray(t)[np.argmax(mask)])

