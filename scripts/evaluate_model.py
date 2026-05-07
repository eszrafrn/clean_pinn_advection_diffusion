import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
root = Path(__file__).resolve().parent.parent
src = root/"src"
sys.path.insert(0, str(src))

from advection_diffusion.metrics import (
    compute_mass,
    conservation_error,
    first_threshold_time,
    linf_error,
    reference_mass_error,
    relative_l2_error,
    total_variation,
)
from advection_diffusion.pinn_model import load_checkpoint
from advection_diffusion.plotting import plot_mass_history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--outdir", default="outputs/evaluation")
    parser.add_argument("--osc-threshold", type=float, default=1e-3)
    return parser.parse_args()


def predict_model(model, x, t, device):
    model.eval()
    x_tensor = torch.tensor(x.reshape(-1, 1), dtype=torch.float32, device=device)
    pred = []
    with torch.no_grad():
        for tv in t:
            t_tensor = torch.full_like(x_tensor, float(tv))
            c = model(torch.cat([x_tensor, t_tensor], dim=1)).detach().cpu().numpy().reshape(-1)
            pred.append(c)
    return np.asarray(pred)


def main():
    args = parse_args()
    ref = np.load(args.reference, allow_pickle=True)
    x = ref["x"]
    t = ref["t"]
    c_cn = ref["c"]
    L = float(ref["L"])
    bc_type = str(ref["bc_type"])
    grid_type = str(ref["grid_type"])
    periodic = bc_type == "periodic"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(args.model, map_location=device)
    model = model.to(device)
    c_pinn = predict_model(model, x, t, device)

    M0 = compute_mass(c_cn[0], x=x, L=L, grid_type=grid_type)
    mass_cn = np.array([compute_mass(c, x=x, L=L, grid_type=grid_type) for c in c_cn])
    mass_pinn = np.array([compute_mass(c, x=x, L=L, grid_type=grid_type) for c in c_pinn])

    cons_cn = np.array([conservation_error(m, M0) for m in mass_cn])
    cons_pinn = np.array([conservation_error(m, M0) for m in mass_pinn])
    ref_mass_err = np.array(
        [reference_mass_error(mp, mc, M0) for mp, mc in zip(mass_pinn, mass_cn)]
    )
    l2_time = np.array([relative_l2_error(cp, cc) for cp, cc in zip(c_pinn, c_cn)])
    linf_time = np.array([linf_error(cp, cc) for cp, cc in zip(c_pinn, c_cn)])
    min_pinn = np.min(c_pinn, axis=1)
    tv_pinn = np.array([total_variation(c, periodic=periodic) for c in c_pinn])
    tv_cn = np.array([total_variation(c, periodic=periodic) for c in c_cn])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.model).stem + "__vs__" + Path(args.reference).stem

    np.savez(
        outdir / f"{tag}_timeseries.npz",
        x=x,
        t=t,
        c_cn=c_cn,
        c_pinn=c_pinn,
        mass_cn=mass_cn,
        mass_pinn=mass_pinn,
        cons_cn=cons_cn,
        cons_pinn=cons_pinn,
        ref_mass_err=ref_mass_err,
        l2_time=l2_time,
        linf_time=linf_time,
        min_pinn=min_pinn,
        tv_pinn=tv_pinn,
        tv_cn=tv_cn,
    )

    plot_mass_history(
        t,
        {"CN": mass_cn, "PINN": mass_pinn},
        save_path=outdir / f"{tag}_mass.png",
        title=f"Mass Evolution: {tag}",
    )

    summary = {
        "model": args.model,
        "reference": args.reference,
        "bc_type": bc_type,
        "M0": float(M0),
        "final_l2": float(l2_time[-1]),
        "max_l2": float(np.max(l2_time)),
        "final_linf": float(linf_time[-1]),
        "final_conservation_error_cn": float(cons_cn[-1]),
        "final_conservation_error_pinn": float(cons_pinn[-1]),
        "max_conservation_error_pinn": float(np.max(cons_pinn)),
        "final_reference_mass_error": float(ref_mass_err[-1]),
        "min_c_pinn": float(np.min(min_pinn)),
        "first_negative_time_threshold": first_threshold_time(t, -min_pinn, args.osc_threshold),
        "max_tv_ratio_pinn_to_initial": float(np.max(tv_pinn) / max(tv_pinn[0], 1e-14)),
    }
    with (outdir / f"{tag}_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

