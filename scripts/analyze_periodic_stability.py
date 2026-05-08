"""
Analyze long-time stability for 1D periodic advection-diffusion runs.

The script compares:
  1. Crank-Nicolson reference stored in a .npz file
  2. Vanilla PINN checkpoint
  3. Conservative / mass-penalty PINN checkpoint

Outputs:
  - raw time-series CSV
  - summary CSV
  - PNG diagnostic plots
  - GIF animation of c(x,t)

Typical usage from the project root:
  python analyze_periodic_stability.py ^
    --reference outputs/reference/reference_periodic_Pe0.25_T5.npz ^
    --vanilla outputs/models/pinn_vanilla_periodic_Pe0.25_T5.pt ^
    --conservative outputs/models/pinn_conservative_periodic_Pe0.25_T5.pt ^
    --pe 0.25 ^
    --outdir outputs/stability/Pe0.25

Batch usage:
  python analyze_periodic_stability.py ^
    --batch ^
    --references-glob "outputs/reference/reference_periodic_Pe*_T5.npz" ^
    --models-dir outputs/models ^
    --outdir outputs/stability
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    return plt, FuncAnimation, PillowWriter


def natural_float(text: str) -> Optional[float]:
    match = re.search(r"Pe([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None


def natural_tmax(text: str) -> Optional[float]:
    match = re.search(r"T([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None


def pe_label(pe: Optional[float]) -> str:
    if pe is None:
        return "unknown"
    return f"{pe:g}".replace(".", "p")


def load_reference_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)

    x = pick_1d(data, ["x", "xs", "x_grid", "x_nodes", "X"])
    t = pick_1d(data, ["t", "time", "times", "t_grid", "T"])
    c = pick_solution(data, len(t), len(x))

    x = np.asarray(x, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    c = np.asarray(c, dtype=float)

    if c.shape == (len(x), len(t)):
        c = c.T
    if c.shape != (len(t), len(x)):
        raise ValueError(
            f"Cannot interpret reference solution shape {c.shape}; "
            f"expected {(len(t), len(x))} or {(len(x), len(t))}."
        )

    return x, t, c


def pick_1d(data: np.lib.npyio.NpzFile, names: Sequence[str]) -> np.ndarray:
    for name in names:
        if name in data and np.asarray(data[name]).ndim == 1:
            return data[name]
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim == 1:
            return arr
    raise KeyError(f"No 1D grid array found. Available keys: {data.files}")


def pick_solution(data: np.lib.npyio.NpzFile, nt: int, nx: int) -> np.ndarray:
    preferred = ["c", "C", "u", "U", "solution", "sol", "reference", "c_cn"]
    for name in preferred:
        if name in data:
            arr = np.asarray(data[name])
            if arr.ndim == 2:
                return arr
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim == 2 and arr.shape in [(nt, nx), (nx, nt)]:
            return arr
    raise KeyError(f"No 2D solution array found. Available keys: {data.files}")


def is_duplicate_endpoint_grid(
    x: np.ndarray,
    values: Optional[np.ndarray] = None,
    grid_type: str = "auto",
) -> bool:
    if grid_type == "endpoint":
        return True
    if grid_type == "periodic-cell":
        return False
    if len(x) < 3:
        return False
    dx = np.diff(x)
    if not np.all(dx > 0):
        return False
    dx_med = float(np.median(dx))
    starts_like_endpoint = abs(float(x[0])) <= 0.25 * dx_med
    if not starts_like_endpoint:
        return False
    if values is None:
        return starts_like_endpoint
    values = np.asarray(values, dtype=float).reshape(-1)
    scale = max(1.0, float(np.max(np.abs(values))))
    return abs(float(values[0] - values[-1])) <= 1e-6 * scale


def domain_length(x: np.ndarray, values: Optional[np.ndarray] = None, grid_type: str = "auto") -> float:
    dx = np.diff(x)
    if len(dx) == 0:
        raise ValueError("x grid must contain at least two points.")
    if is_duplicate_endpoint_grid(x, values, grid_type):
        return float(x[-1] - x[0])
    return float(np.median(dx) * len(x))


def integrate_periodic(x: np.ndarray, values: np.ndarray, grid_type: str = "auto") -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(x) != len(values):
        raise ValueError("x and values must have the same length.")

    length = domain_length(x, values, grid_type)
    if length <= 0:
        raise ValueError("x grid must be increasing.")

    # Node grid including both x=0 and x=L: trapezoid is appropriate.
    # Cell-centered periodic grid: sum with dx=L/N avoids inventing endpoints.
    if is_duplicate_endpoint_grid(x, values, grid_type):
        return float(np.trapz(values, x))
    return float(length / len(x) * np.sum(values))


def l2_error(x: np.ndarray, pred: np.ndarray, ref: np.ndarray, grid_type: str = "auto") -> float:
    err2 = (np.asarray(pred) - np.asarray(ref)) ** 2
    length = domain_length(x, err2, grid_type)
    if length <= 0:
        return float(np.sqrt(np.mean(err2)))
    return float(np.sqrt(integrate_periodic(x, err2, grid_type) / length))


def linf_error(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(pred) - np.asarray(ref))))


def total_variation_periodic(c: np.ndarray, endpoint_grid: bool) -> float:
    c = np.asarray(c, dtype=float).reshape(-1)
    tv = float(np.sum(np.abs(np.diff(c))))
    if not endpoint_grid:
        tv += float(abs(c[0] - c[-1]))
    return tv


def extrema_count(c: np.ndarray) -> int:
    c = np.asarray(c, dtype=float).reshape(-1)
    if c.size < 5:
        return 0
    slope = np.diff(c)
    eps = 1e-13 * max(1.0, float(np.max(np.abs(c))))
    slope[np.abs(slope) < eps] = 0.0
    signs = np.sign(slope)
    signs = signs[signs != 0.0]
    if signs.size < 2:
        return 0
    return int(np.sum(signs[1:] * signs[:-1] < 0.0))


@dataclass
class MethodSeries:
    name: str
    c: np.ndarray


@dataclass
class StabilityThresholds:
    mass_tol: float = 1e-2
    tv_ratio_tol: float = 1.05
    negative_tol_rel: float = 1e-3
    overshoot_tol_rel: float = 1e-3
    l2_tol: Optional[float] = None


def first_time(times: np.ndarray, mask: np.ndarray) -> Optional[float]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    return float(times[int(idx[0])])


def summarize_method(
    rows: List[Dict[str, float | str]],
    method: str,
    times: np.ndarray,
    thresholds: StabilityThresholds,
) -> Dict[str, float | str | None]:
    mr = [r for r in rows if r["method"] == method]
    if not mr:
        raise ValueError(f"No rows for method {method}")

    mass_error = np.array([float(r["conservation_error"]) for r in mr])
    l2 = np.array([float(r["l2_vs_cn"]) for r in mr])
    min_c = np.array([float(r["min_c"]) for r in mr])
    tv_ratio = np.array([float(r["tv_ratio"]) for r in mr])
    overshoot = np.array([float(r["overshoot_rel"]) for r in mr])
    neg_depth = np.array([float(r["negative_depth_rel"]) for r in mr])

    first_mass = first_time(times, mass_error > thresholds.mass_tol)
    first_tv = first_time(times, tv_ratio > thresholds.tv_ratio_tol)
    first_negative = first_time(times, neg_depth > thresholds.negative_tol_rel)
    first_overshoot = first_time(times, overshoot > thresholds.overshoot_tol_rel)
    first_l2 = None
    if thresholds.l2_tol is not None:
        first_l2 = first_time(times, l2 > thresholds.l2_tol)

    candidates = [
        ("mass", first_mass),
        ("tv_growth", first_tv),
        ("negative", first_negative),
        ("overshoot", first_overshoot),
        ("l2", first_l2),
    ]
    finite = [(name, value) for name, value in candidates if value is not None]
    if finite:
        unstable_time = min(value for _, value in finite)
        triggers = ",".join(name for name, value in finite if value == unstable_time)
    else:
        unstable_time = None
        triggers = ""

    return {
        "method": method,
        "final_l2_vs_cn": float(l2[-1]),
        "max_l2_vs_cn": float(np.max(l2)),
        "final_conservation_error": float(mass_error[-1]),
        "max_conservation_error": float(np.max(mass_error)),
        "min_c": float(np.min(min_c)),
        "max_tv_ratio": float(np.max(tv_ratio)),
        "max_overshoot_rel": float(np.max(overshoot)),
        "max_negative_depth_rel": float(np.max(neg_depth)),
        "first_mass_error_time": first_mass,
        "first_tv_growth_time": first_tv,
        "first_negative_time": first_negative,
        "first_overshoot_time": first_overshoot,
        "first_l2_error_time": first_l2,
        "unstable_time": unstable_time,
        "unstable_trigger": triggers,
    }


def build_rows(
    x: np.ndarray,
    t: np.ndarray,
    c_ref: np.ndarray,
    methods: Sequence[MethodSeries],
    thresholds: StabilityThresholds,
    grid_type: str,
) -> Tuple[List[Dict[str, float | str]], List[Dict[str, float | str | None]]]:
    endpoint = is_duplicate_endpoint_grid(x, c_ref[0], grid_type)
    m0 = integrate_periodic(x, c_ref[0], grid_type)
    tv0 = total_variation_periodic(c_ref[0], endpoint)
    c0_min = float(np.min(c_ref[0]))
    c0_max = float(np.max(c_ref[0]))
    c0_scale = max(abs(c0_max), abs(c0_min), 1e-12)

    rows: List[Dict[str, float | str]] = []
    for series in methods:
        if series.c.shape != c_ref.shape:
            raise ValueError(
                f"{series.name} shape {series.c.shape} does not match reference {c_ref.shape}."
            )
        for k, tk in enumerate(t):
            ck = series.c[k]
            mass = integrate_periodic(x, ck, grid_type)
            min_c = float(np.min(ck))
            max_c = float(np.max(ck))
            tv = total_variation_periodic(ck, endpoint)
            row = {
                "time": float(tk),
                "method": series.name,
                "mass": mass,
                "conservation_error": abs(mass - m0) / max(abs(m0), 1e-12),
                "l2_vs_cn": 0.0 if series.name == "CN" else l2_error(x, ck, c_ref[k], grid_type),
                "linf_vs_cn": 0.0 if series.name == "CN" else linf_error(ck, c_ref[k]),
                "min_c": min_c,
                "max_c": max_c,
                "tv": tv,
                "tv_ratio": tv / max(tv0, 1e-12),
                "overshoot_rel": max(0.0, max_c - c0_max) / c0_scale,
                "negative_depth_rel": max(0.0, -min_c) / c0_scale,
                "extrema_count": extrema_count(ck),
            }
            rows.append(row)

    summaries = [
        summarize_method(rows, series.name, t, thresholds)
        for series in methods
    ]
    return rows, summaries


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_rows_by_method(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append(row)
    return grouped


def plot_diagnostics(
    x: np.ndarray,
    t: np.ndarray,
    c_ref: np.ndarray,
    methods: Sequence[MethodSeries],
    rows: Sequence[Dict[str, object]],
    outdir: Path,
    label: str,
) -> None:
    plt, _, _ = import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)
    grouped = read_rows_by_method(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    for method, method_rows in grouped.items():
        ax.plot(
            [float(r["time"]) for r in method_rows],
            [float(r["conservation_error"]) for r in method_rows],
            label=method,
        )
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("relative mass error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    for method, method_rows in grouped.items():
        if method == "CN":
            continue
        ax.plot(
            [float(r["time"]) for r in method_rows],
            [float(r["l2_vs_cn"]) for r in method_rows],
            label=method,
        )
    ax.set_xlabel("t")
    ax.set_ylabel("L2 error vs CN")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    for method, method_rows in grouped.items():
        ax.plot(
            [float(r["time"]) for r in method_rows],
            [float(r["tv_ratio"]) for r in method_rows],
            label=method,
        )
    ax.axhline(1.0, color="k", lw=1, alpha=0.4)
    ax.set_xlabel("t")
    ax.set_ylabel("TV(t) / TV(0)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    for method, method_rows in grouped.items():
        ax.plot(
            [float(r["time"]) for r in method_rows],
            [float(r["min_c"]) for r in method_rows],
            label=method,
        )
    ax.axhline(0.0, color="k", lw=1, alpha=0.4)
    ax.set_xlabel("t")
    ax.set_ylabel("minimum concentration")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(f"Periodic stability diagnostics - Pe {label}")
    fig.savefig(outdir / f"stability_diagnostics_Pe{label}.png", dpi=180)
    plt.close(fig)

    snapshot_times = np.linspace(float(t[0]), float(t[-1]), 6)
    snapshot_indices = [int(np.argmin(np.abs(t - ts))) for ts in snapshot_times]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.ravel()
    for ax, idx in zip(axes_flat, snapshot_indices):
        for series in methods:
            ax.plot(x, series.c[idx], label=series.name, lw=1.8)
        ax.set_title(f"t = {t[idx]:.3g}")
        ax.grid(True, alpha=0.25)
    axes_flat[0].legend()
    for ax in axes_flat[3:]:
        ax.set_xlabel("x")
    for ax in axes_flat[::3]:
        ax.set_ylabel("c")
    fig.suptitle(f"Solution snapshots - Pe {label}")
    fig.savefig(outdir / f"solution_snapshots_Pe{label}.png", dpi=180)
    plt.close(fig)


def make_animation(
    x: np.ndarray,
    t: np.ndarray,
    methods: Sequence[MethodSeries],
    outdir: Path,
    label: str,
    fps: int = 12,
    max_frames: int = 160,
) -> None:
    plt, FuncAnimation, PillowWriter = import_matplotlib()
    outdir.mkdir(parents=True, exist_ok=True)

    nframes = min(max_frames, len(t))
    frame_indices = np.linspace(0, len(t) - 1, nframes).astype(int)

    all_values = np.concatenate([series.c.reshape(-1) for series in methods])
    ymin = float(np.min(all_values))
    ymax = float(np.max(all_values))
    pad = 0.08 * max(ymax - ymin, 1e-12)

    fig, ax = plt.subplots(figsize=(9, 5))
    lines = []
    for series in methods:
        (line,) = ax.plot([], [], lw=2, label=series.name)
        lines.append(line)
    title = ax.set_title("")
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("x")
    ax.set_ylabel("c(x,t)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    def init():
        for line in lines:
            line.set_data([], [])
        return (*lines, title)

    def update(frame_number: int):
        idx = frame_indices[frame_number]
        for line, series in zip(lines, methods):
            line.set_data(x, series.c[idx])
        title.set_text(f"Periodic ADE stability - Pe {label}, t = {t[idx]:.3g}")
        return (*lines, title)

    animation = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frame_indices),
        interval=1000 / max(fps, 1),
        blit=True,
    )
    animation.save(outdir / f"solution_animation_Pe{label}.gif", writer=PillowWriter(fps=fps))
    plt.close(fig)


class FallbackPINN:
    def __init__(self, checkpoint: Path, device: str, length: float, tmax: float):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to evaluate PINN checkpoints.") from exc

        self.torch = torch
        self.device = device
        payload = torch.load(checkpoint, map_location=device)
        if isinstance(payload, nn.Module):
            self.model = payload.to(device)
            self.model.eval()
            self.length = length
            self.tmax = tmax
            return

        state = extract_state_dict(payload)
        state = normalize_state_dict(state)
        dims = infer_sequential_dims(state)
        model = GenericSequentialPINN(dims=dims, length=length, tmax=tmax, nn=nn)
        missing, unexpected = model.load_state_dict(state, strict=False)
        linear_missing = [k for k in missing if k.endswith(".weight") or k.endswith(".bias")]
        linear_unexpected = [k for k in unexpected if k.endswith(".weight") or k.endswith(".bias")]
        if linear_missing or linear_unexpected:
            raise RuntimeError(
                "Could not load checkpoint into fallback model. "
                f"Missing={linear_missing}, unexpected={linear_unexpected}. "
                "Run this script inside the project root or adapt load_model_from_project()."
            )
        self.model = model.to(device)
        self.model.eval()
        self.length = length
        self.tmax = tmax

    def predict(self, x: np.ndarray, t: np.ndarray, chunk: int = 65536) -> np.ndarray:
        torch = self.torch
        out = np.empty((len(t), len(x)), dtype=float)
        xcol_base = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            for k, tk in enumerate(t):
                tcol_base = torch.full_like(xcol_base, float(tk))
                values = []
                for start in range(0, len(x), chunk):
                    xs = xcol_base[start : start + chunk]
                    ts = tcol_base[start : start + chunk]
                    pred = self.model(xs, ts)
                    values.append(pred.detach().cpu().numpy().reshape(-1))
                out[k] = np.concatenate(values)
        return out


class GenericSequentialPINN:
    def __init__(self, dims: Sequence[int], length: float, tmax: float, nn):
        super().__init__()
        self.nn = nn
        self.length = float(length)
        self.tmax = float(tmax)
        modules = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)

    def __call__(self, x, t):
        return self.forward(x, t)

    def to(self, device):
        self.net.to(device)
        return self

    def eval(self):
        self.net.eval()
        return self

    def load_state_dict(self, state, strict=False):
        return self.net.load_state_dict(state, strict=strict)

    def forward(self, x, t):
        torch = __import__("torch")
        x_scaled = 2.0 * x / self.length - 1.0
        t_scaled = 2.0 * t / self.tmax - 1.0
        xt = torch.cat([x_scaled, t_scaled], dim=1)
        return self.net(xt)


def extract_state_dict(payload) -> Dict[str, object]:
    if isinstance(payload, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        if all(hasattr(v, "shape") for v in payload.values()):
            return payload
    raise RuntimeError(
        "Unsupported checkpoint format. Expected a torch nn.Module, state_dict, "
        "or dict containing model_state_dict/state_dict."
    )


def normalize_state_dict(state: Dict[str, object]) -> Dict[str, object]:
    keys = list(state.keys())
    prefixes = ["module.", "model.", "pinn.", "network.", "net."]
    stripped = dict(state)
    changed = True
    while changed:
        changed = False
        keys = list(stripped.keys())
        for prefix in prefixes:
            if keys and all(k.startswith(prefix) for k in keys):
                stripped = {k[len(prefix) :]: v for k, v in stripped.items()}
                changed = True
                break

    linear = sorted(
        (k for k in stripped if re.fullmatch(r"\d+\.weight", k)),
        key=lambda k: int(k.split(".")[0]),
    )
    if not linear:
        # Try keys like layers.0.weight, linears.0.weight, or mlp.0.weight.
        candidates = []
        for key in stripped:
            match = re.search(r"(\d+)\.weight$", key)
            value = stripped[key]
            if match and getattr(value, "ndim", None) == 2:
                candidates.append((int(match.group(1)), key))
        if not candidates:
            raise RuntimeError(f"Could not infer linear layers from keys: {list(state.keys())[:8]}")
        candidates.sort()
        remap = {}
        for new_idx, (_, old_weight_key) in enumerate(candidates):
            old_prefix = old_weight_key.rsplit(".", 1)[0]
            seq_idx = 2 * new_idx
            remap[f"{seq_idx}.weight"] = stripped[old_weight_key]
            bias_key = old_prefix + ".bias"
            if bias_key in stripped:
                remap[f"{seq_idx}.bias"] = stripped[bias_key]
        return remap

    return stripped


def infer_sequential_dims(state: Dict[str, object]) -> List[int]:
    weight_keys = sorted(
        (k for k in state if re.fullmatch(r"\d+\.weight", k)),
        key=lambda k: int(k.split(".")[0]),
    )
    if not weight_keys:
        raise RuntimeError("No sequential Linear weight keys found.")
    dims: List[int] = []
    for i, key in enumerate(weight_keys):
        weight = state[key]
        shape = tuple(int(v) for v in weight.shape)
        if len(shape) != 2:
            continue
        out_features, in_features = shape
        if i == 0:
            dims.append(in_features)
        dims.append(out_features)
    if dims[0] != 2:
        raise RuntimeError(f"Expected PINN input dimension 2 for (x,t), got {dims[0]}.")
    return dims


def load_model_from_project(
    checkpoint: Path,
    project_root: Path,
    device: str,
    length: float,
    tmax: float,
) -> Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    src = project_root / "src"
    if src.exists():
        sys.path.insert(0, str(src))
    sys.path.insert(0, str(project_root))

    # If the project already has a checkpoint loader, use it. Names are tried
    # conservatively to avoid depending on one exact implementation.
    try:
        import importlib

        module = importlib.import_module("advection_diffusion.pinn_model")
    except Exception:
        return None

    loader = None
    for name in ["load_pinn_checkpoint", "load_checkpoint", "load_model"]:
        if hasattr(module, name):
            loader = getattr(module, name)
            break
    if loader is None:
        return None

    try:
        model = loader(checkpoint, device=device)
    except TypeError:
        try:
            model = loader(checkpoint)
        except Exception:
            return None
    except Exception:
        return None

    try:
        import torch
    except ImportError:
        return None

    if isinstance(model, tuple):
        model = model[0]
    model = model.to(device)
    model.eval()

    def predict(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        out = np.empty((len(t), len(x)), dtype=float)
        xcol = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=device)
        with torch.no_grad():
            for k, tk in enumerate(t):
                tcol = torch.full_like(xcol, float(tk))
                pred = model(xcol, tcol)
                out[k] = pred.detach().cpu().numpy().reshape(-1)
        return out

    return predict


def predict_checkpoint(
    checkpoint: Path,
    x: np.ndarray,
    t: np.ndarray,
    project_root: Path,
    device: str,
) -> np.ndarray:
    length = float(x[-1] - x[0])
    tmax = float(t[-1])
    project_predictor = load_model_from_project(checkpoint, project_root, device, length, tmax)
    if project_predictor is not None:
        return project_predictor(x, t)
    return FallbackPINN(checkpoint, device, length, tmax).predict(x, t)


def analyze_one(args, reference: Path, vanilla: Optional[Path], conservative: Optional[Path], pe: Optional[float]) -> None:
    x, t, c_ref = load_reference_npz(reference)
    label = pe_label(pe)
    outdir = Path(args.outdir)
    if args.batch:
        outdir = outdir / f"Pe{label}"
    outdir.mkdir(parents=True, exist_ok=True)

    methods: List[MethodSeries] = [MethodSeries("CN", c_ref)]

    if vanilla is not None:
        c_vanilla = predict_checkpoint(vanilla, x, t, Path(args.project_root), args.device)
        methods.append(MethodSeries("PINN Vanilla", c_vanilla))

    if conservative is not None:
        c_cons = predict_checkpoint(conservative, x, t, Path(args.project_root), args.device)
        methods.append(MethodSeries("PINN Conservative", c_cons))

    thresholds = StabilityThresholds(
        mass_tol=args.mass_tol,
        tv_ratio_tol=args.tv_ratio_tol,
        negative_tol_rel=args.negative_tol_rel,
        overshoot_tol_rel=args.overshoot_tol_rel,
        l2_tol=args.l2_tol,
    )
    rows, summaries = build_rows(x, t, c_ref, methods, thresholds, args.grid_type)

    raw_csv = outdir / f"stability_timeseries_Pe{label}.csv"
    summary_csv = outdir / f"stability_summary_Pe{label}.csv"
    write_csv(raw_csv, rows)
    write_csv(summary_csv, summaries)

    with (outdir / f"stability_config_Pe{label}.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "reference": str(reference),
                "vanilla": str(vanilla) if vanilla else None,
                "conservative": str(conservative) if conservative else None,
                "pe": pe,
                "thresholds": thresholds.__dict__,
                "grid_type": args.grid_type,
            },
            handle,
            indent=2,
        )

    if not args.no_plots:
        plot_diagnostics(x, t, c_ref, methods, rows, outdir, label)
    if args.animate:
        make_animation(x, t, methods, outdir, label, fps=args.fps, max_frames=args.max_frames)

    print(f"[OK] Pe={pe if pe is not None else 'unknown'}")
    print(f"  raw data : {raw_csv}")
    print(f"  summary  : {summary_csv}")
    if not args.no_plots:
        print(f"  plots    : {outdir}")
    if args.animate:
        print(f"  gif      : {outdir / f'solution_animation_Pe{label}.gif'}")


def find_model(models_dir: Path, kind: str, pe: Optional[float], tmax: Optional[float]) -> Optional[Path]:
    if pe is None:
        return None
    pe_text = f"Pe{pe:g}"
    candidates = list(models_dir.glob(f"pinn_{kind}_periodic_{pe_text}_T*.pt"))
    if not candidates:
        candidates = list(models_dir.glob(f"*{kind}*periodic*{pe_text}*.pt"))
    if tmax is not None:
        exact = [p for p in candidates if f"T{tmax:g}" in p.name]
        if exact:
            candidates = exact
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (len(p.name), p.name))[0]


def run_batch(args) -> None:
    refs = sorted(Path().glob(args.references_glob))
    if not refs:
        refs = sorted(Path(args.project_root).glob(args.references_glob))
    if not refs:
        raise FileNotFoundError(f"No references matched {args.references_glob}")

    models_dir = Path(args.models_dir)
    for ref in refs:
        pe = natural_float(ref.name)
        tmax = natural_tmax(ref.name)
        vanilla = find_model(models_dir, "vanilla", pe, tmax)
        conservative = find_model(models_dir, "conservative", pe, tmax)
        if vanilla is None:
            print(f"[WARN] Missing vanilla model for {ref.name}")
        if conservative is None:
            print(f"[WARN] Missing conservative model for {ref.name}")
        analyze_one(args, ref, vanilla, conservative, pe)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", action="store_true", help="Analyze all references matched by --references-glob.")
    parser.add_argument("--reference", type=Path, help="CN reference .npz for one Pe.")
    parser.add_argument("--vanilla", type=Path, help="Vanilla PINN checkpoint .pt.")
    parser.add_argument("--conservative", type=Path, help="Conservative PINN checkpoint .pt.")
    parser.add_argument("--pe", type=float, default=None, help="Peclet number label for output filenames.")
    parser.add_argument("--references-glob", default="outputs/reference/reference_periodic_Pe*_T5.npz")
    parser.add_argument("--models-dir", default="outputs/models")
    parser.add_argument("--project-root", default=".", help="Project root containing src/ if available.")
    parser.add_argument("--outdir", default="outputs/stability")
    parser.add_argument("--device", default="cpu", help="cpu or cuda.")
    parser.add_argument(
        "--grid-type",
        default="auto",
        choices=["auto", "endpoint", "periodic-cell"],
        help=(
            "endpoint: x includes both 0 and L; periodic-cell: x excludes duplicated endpoint; "
            "auto: infer from x and c(0)."
        ),
    )
    parser.add_argument("--mass-tol", type=float, default=1e-2, help="Relative mass-error threshold.")
    parser.add_argument("--tv-ratio-tol", type=float, default=1.05, help="TV growth threshold.")
    parser.add_argument("--negative-tol-rel", type=float, default=1e-3, help="Relative negative undershoot threshold.")
    parser.add_argument("--overshoot-tol-rel", type=float, default=1e-3, help="Relative overshoot threshold.")
    parser.add_argument("--l2-tol", type=float, default=None, help="Optional L2-vs-CN instability threshold.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--animate", action="store_true", help="Also save GIF animation.")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-frames", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch:
        run_batch(args)
        return
    if args.reference is None:
        raise SystemExit("--reference is required unless --batch is used.")
    pe = args.pe if args.pe is not None else natural_float(args.reference.name)
    analyze_one(args, args.reference, args.vanilla, args.conservative, pe)


if __name__ == "__main__":
    main()
