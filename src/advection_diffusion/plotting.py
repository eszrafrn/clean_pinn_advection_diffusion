from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def plot_mass_history(t, mass_dict, save_path=None, title="Mass Evolution"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, mass in mass_dict.items():
        ax.plot(t, mass, label=label)
    first_mass = next(iter(mass_dict.values()))
    ax.axhline(first_mass[0], color="black", linestyle="--", linewidth=1, label="M0")
    ax.set_xlabel("t")
    ax.set_ylabel("Mass")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=250)
    return fig


def animate_cn_vs_pinn(x, t, c_cn, c_pinn=None, save_path="outputs/animations/solution.gif"):
    x = np.asarray(x).reshape(-1)
    t = np.asarray(t).reshape(-1)
    c_cn = np.asarray(c_cn)
    if c_pinn is not None:
        c_pinn = np.asarray(c_pinn)

    y_min = float(np.min(c_cn))
    y_max = float(np.max(c_cn))
    if c_pinn is not None:
        y_min = min(y_min, float(np.min(c_pinn)))
        y_max = max(y_max, float(np.max(c_pinn)))
    margin = 0.1 * max(y_max - y_min, 1e-8)

    fig, ax = plt.subplots(figsize=(8, 5))
    line_cn, = ax.plot([], [], color="navy", linewidth=2.2, label="CN reference")
    line_pinn = None
    if c_pinn is not None:
        line_pinn, = ax.plot([], [], color="crimson", linestyle="--", linewidth=2.0, label="PINN")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel("x")
    ax.set_ylabel("c(x,t)")
    ax.grid(True, alpha=0.35)
    ax.legend()

    def init():
        line_cn.set_data([], [])
        if line_pinn is not None:
            line_pinn.set_data([], [])
        time_text.set_text("")
        return tuple(v for v in [line_cn, line_pinn, time_text] if v is not None)

    def update(frame):
        line_cn.set_data(x, c_cn[frame])
        if line_pinn is not None:
            line_pinn.set_data(x, c_pinn[frame])
        time_text.set_text(f"t = {t[frame]:.4f}")
        return tuple(v for v in [line_cn, line_pinn, time_text] if v is not None)

    step = max(1, len(t) // 250)
    frames = list(range(0, len(t), step))
    if frames[-1] != len(t) - 1:
        frames.append(len(t) - 1)

    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer=PillowWriter(fps=24))
    plt.close(fig)
    return save_path

